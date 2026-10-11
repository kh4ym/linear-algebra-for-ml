/* p5-04 — L1 vs L2: why one gives zeros (regularization as a norm ball meeting loss contours). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, C = LA.color;
  const TAU = Math.PI * 2;
  const LEVELS = [0.1, 0.3, 0.6, 1, 1.5, 2.2];
  const WS0 = [2, 1];
  const ws = '<span class="c-gold">w*</span>';

  /* ---------- pure helpers ---------- */
  const getH = s => [[1, s.rho], [s.rho, 1]];
  function loss(H, c, w) { const d = [w[0] - c[0], w[1] - c[1]]; return 0.5 * (d[0] * (H[0][0] * d[0] + H[0][1] * d[1]) + d[1] * (H[1][0] * d[0] + H[1][1] * d[1])); }
  function grad(H, c, w) { return N.matVec(H, N.sub(w, c)); }
  /** point of the Lp ball boundary (radius t) at angle th */
  function lpPoint(p, t, th) { const c = Math.cos(th), s = Math.sin(th); const n = Math.pow(Math.pow(Math.abs(c), p) + Math.pow(Math.abs(s), p), 1 / p) || 1; return [t * c / n, t * s / n]; }
  function ballPts(p, t, n) { if (Math.abs(p - 1) < 1e-9) return [[t, 0], [0, t], [-t, 0], [0, -t]]; const pts = []; for (let i = 0; i < n; i++) pts.push(lpPoint(p, t, i / n * TAU)); return pts; }
  /** argmin of the loss over the Lp ball of radius t (720 boundary samples incl. the axis angles) */
  function solveBall(H, c, p, t) {
    if (t <= 1e-6) return [0, 0];
    if (N.normP(c, p) <= t) return c.slice();
    let best = [t, 0], bv = Infinity;
    for (let i = 0; i < 720; i++) { const w = lpPoint(p, t, i / 720 * TAU); const v = loss(H, c, w); if (v < bv) { bv = v; best = w; } }
    return best;
  }
  function ridgeSol(H, c, lam) { const A = [[H[0][0] + lam, H[0][1]], [H[1][0], H[1][1] + lam]]; return N.solve(A, N.matVec(H, c)) || c.slice(); }
  const memo = new Map();
  /** 60 constrained solutions for t from 0 to ‖w*‖_p (memoized; pure in its arguments) */
  function pathFor(H, c, p) {
    const key = p + '|' + H[0][1].toFixed(3) + '|' + c[0].toFixed(3) + ',' + c[1].toFixed(3);
    const hit = memo.get(key); if (hit) return hit;
    const T = N.normP(c, p), pts = [];
    for (let k = 0; k < 60; k++) pts.push({ t: T * k / 59, w: solveBall(H, c, p, T * k / 59) });
    if (memo.size > 48) memo.clear();
    memo.set(key, pts); return pts;
  }
  /** largest budget t at which the L1 solution still sits on the w₁ axis (corner contact) */
  function kinkT(H, c) {
    const sx = c[0] >= 0 ? 1 : -1; let best = 0;
    for (let i = 0; i <= 400; i++) { const t = Math.abs(c[0]) * i / 400; const g = N.scale(grad(H, c, [sx * t, 0]), -1); if (g[0] * sx > 0 && Math.abs(g[1]) <= Math.abs(g[0])) best = t; }
    return best;
  }
  /** matrix whose unit-circle image is the level-c contour of ½(w−w*)ᵀH(w−w*) */
  function ellipseM(H, c) {
    const e = N.eigSym(H); const q1 = e.vectors[0], q2 = e.vectors[1];
    const a = Math.sqrt(2 * c / Math.max(e.values[0], 0.03)), b = Math.sqrt(2 * c / Math.max(e.values[1], 0.03));
    return [[q1[0] * a, q2[0] * b], [q1[1] * a, q2[1] * b]];
  }
  function gdPath(H, c) { let w = [-0.2, 2.7]; const out = [w]; for (let k = 0; k < 14; k++) { w = N.sub(w, N.scale(grad(H, c, w), 0.55)); out.push(w); } return out; }
  function currentSol(s) { const H = getH(s); return s.mode === 'ridge' ? ridgeSol(H, s.wstar, s.lambda) : solveBall(H, s.wstar, s.p, s.t); }
  /** 12 synthetic weights and their lasso (soft-threshold) version with exactly 7 zeros */
  function stripData() {
    const rand = LA.rng(504); const before = []; for (let i = 0; i < 12; i++) before.push(LA.rng.gauss(rand) * 1.1);
    const abs = before.map(Math.abs).sort((a, b) => a - b); const a = abs[6] + 1e-9;
    const after = before.map(x => Math.sign(x) * Math.max(Math.abs(x) - a, 0));
    return { before, after };
  }
  const divRamp = t => C.diverge(t * 2 - 1);

  LA.registerLesson({
    id: 'p5-04-regularization-geometry',
    part: 5,
    title: 'L1 vs L2: why one gives zeros',
    subtitle: 'A norm ball meets the loss contours; a corner on an axis makes a weight exactly zero.',
    minutes: 10,
    prereqs: ['p1-05-norms-distance', 'p5-03-gradient-descent', 'p4-12-pseudoinverse-conditioning'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D (two weights)', realSize: 'lasso over 10⁴ gene features; weight decay on 10⁹ weights', transfers: 'solution = first contact between loss contours and a norm ball; corners on axes give exact zeros; ridge = (H + λI)⁻¹Hw* — exact in any n; the L1 ball in n-D has 2n corners, so the cartoon understates sparsity' },
    cantShow: ['the 2n corners and 2ⁿ faces of the L1 ball in n-D', 'elastic net and group norms (one sentence in the recap only)', 'non-quadratic losses (logistic) whose contours are not ellipses'],
    terms: [
      { term: 'regularization', def: 'Adding a penalty on the size of the weights to the loss, or equivalently limiting the weights to a ball.' },
      { term: 'norm ball', def: 'All weight vectors w with ‖w‖ ≤ t for some norm: a disk for L2, a diamond for L1.' },
      { term: 'ridge', def: 'The L2 penalty λ‖w‖₂²; its ball is the disk and its solution is (H + λI)⁻¹Hw*.' },
      { term: 'lasso', def: 'The L1 penalty λ‖w‖₁; its ball is the diamond, whose corners sit on the axes.' },
      { term: 'sparsity', def: 'Some weights being exactly zero, not merely small.' },
      { term: 'solution path', def: 'The curve traced by the regularized solution as the budget t (or the penalty λ) sweeps.' },
      { term: 'weight decay', def: 'The L2 penalty written as a per-step shrink: multiply w by (1 − ηλ) before each gradient step.' },
    ],
    atlas: ['regularization', 'linear-regression', 'optimization'],
    thumbnail: 6,

    state() {
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), { center: [1.15, 1.05], halfH: 2.6 }),
        rho: 0.5, wstar: WS0.slice(), p: 2, t: 0, lambda: 0, mode: 'ball',
        showBall: false, showSol: false, showPaths: false, showBoth: false, readouts: false,
        gd: 0, tangentA: 0, gradA: 0, fanA: 0, fanAt: 1, diskCmpA: 0, crossA: 0, glow: false,
        lift: -1, mlTie: false, mlA: 0, wdK: 0, recap: false,
        guess: [0, 0], showGuess: false, mathMode: 'loss', noBudget: false,
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g, s); return; }
      if (s.mlTie) { renderML(g, s); return; }
      const H = getH(s), c = s.wstar;
      g.ghostGrid({ range: 6 });
      g.axes({ names: ['w₁', 'w₂'] });
      for (const lv of LEVELS) g.ellipse(ellipseM(H, lv), c, { stroke: 'residual', width: 1.1, strokeAlpha: 0.5, segments: 96 });
      if (s.gd > 0) {
        const path = gdPath(H, c); const n = Math.max(1, Math.round(s.gd * (path.length - 1)));
        g.polyline(path.slice(0, n + 1), { color: 'b', width: 1.6, alpha: 0.55 });
        g.points(path.slice(0, n + 1), { r: 2.5, color: 'b', alpha: 0.6 });
      }
      const sol = currentSol(s);
      const tBall = s.mode === 'ridge' ? N.norm(sol) : s.t;
      // check 3 (showGuess) hides the ball, the ridge trail and the true point: each would show the answer
      if (s.showBall && tBall > 1e-3 && !s.showGuess) {
        if (s.mode === 'ridge' || Math.abs(s.p - 2) < 1e-9) g.circle([0, 0], tBall, { fill: 'teal', alpha: 0.2, stroke: 'teal', width: 1.6 });
        else g.polygon(ballPts(s.p, tBall, 160), { fill: 'teal', alpha: 0.2, stroke: 'teal', width: 1.6 });
      }
      if (s.mode === 'ridge' && s.lambda > 1e-6 && !s.showGuess) {
        const trail = []; for (let k = 0; k < 40; k++) trail.push(ridgeSol(H, c, s.lambda * k / 39));
        g.polyline(trail, { color: 'b', width: 2.2, dashed: true, alpha: 0.95 });
        g.label('ridge path', trail[Math.min(8, trail.length - 1)], { color: 'b', size: 12, dy: -14, dx: -10 });
      }
      if (s.showBoth) {
        g.circle([0, 0], s.t, { fill: null, stroke: 'teal', width: 1.2, dashed: true, strokeAlpha: 0.55 });
        g.polygon(ballPts(1, s.t, 4), { fill: 'teal', alpha: 0.08, stroke: 'teal', width: 1.2 });
      }
      if (s.showPaths) {
        const P1 = pathFor(H, c, 1), P2 = pathFor(H, c, 2);
        const draw = (P, p, dashed) => {
          const pts = P.filter(q => q.t <= s.t + 1e-9).map(q => q.w);
          pts.push(solveBall(H, c, p, s.t));
          if (pts.length > 1) g.polyline(pts, { color: 'b', width: 2.2, dashed, alpha: 0.95 });
        };
        draw(P1, 1, false); draw(P2, 2, true);
        const kt = kinkT(H, c), sx = c[0] >= 0 ? 1 : -1;
        if (s.t >= kt - 1e-6 && kt > 0.02) { const k = 0.05; g.polygon([[sx * kt - k, -k], [sx * kt + k, -k], [sx * kt + k, k], [sx * kt - k, k]], { fill: 'b', alpha: 0.9, stroke: 'b' }); g.label('kink', [sx * kt, 0], { color: 'b', anchor: 'below', dy: 10, size: 12 }); }
      }
      // gold contour through the current solution
      const lv = loss(H, c, sol);
      if (s.showSol && lv > 1e-4) g.ellipse(ellipseM(H, lv), c, { stroke: 'gold', width: 1.8, segments: 96 });
      // tangent, gradient and w at the touch (step 3)
      if (s.tangentA > 0 && N.norm(sol) > 1e-6) {
        const u = N.normalize(sol), d = [-u[1], u[0]];
        g.line(N.add(sol, N.scale(d, -1.3)), N.add(sol, N.scale(d, 1.3)), { color: 'residual', dashed: true, width: 1.4, alpha: s.tangentA });
        g.arc(sol, d, u, { r: 0.22, right: true, color: 'hud' });
      }
      if (s.gradA > 0) {
        g.vector([0, 0], sol, { color: 'w', label: 'w', labelAt: 0.3, width: 2.8, alpha: s.gradA });
        const gr = grad(H, c, sol);
        g.vector(sol, N.add(sol, N.scale(gr, 0.25)), { color: 'v', label: '∇L', width: 2.6, alpha: s.gradA, labelOffset: -16 });
      }
      // the corner fan (steps 6 and 9)
      if (s.fanA > 0) {
        const sx = c[0] >= 0 ? 1 : -1, corner = [sx * s.fanAt, 0];
        for (let i = 0; i <= 8; i++) {
          const a = (-45 + 90 * i / 8) * Math.PI / 180, edge = i === 0 || i === 8;
          g.vector(corner, [corner[0] + sx * 0.6 * Math.cos(a), corner[1] + 0.6 * Math.sin(a)], { color: edge ? 'ok' : 'hud', width: edge ? 2 : 1.2, head: 7, alpha: (edge ? 0.95 : 0.3) * s.fanA });
        }
        g.label('outward fan', [corner[0] + sx * 0.3, -0.62], { color: 'muted', size: 12, anchor: sx > 0 ? 'left' : 'right', alpha: s.fanA });
        const ng = N.scale(grad(H, c, corner), -0.4);
        g.vector(corner, N.add(corner, ng), { color: 'v', label: '−∇L', width: 2.8, alpha: s.fanA });
      }
      if (s.diskCmpA > 0) {
        const s2 = solveBall(H, c, 2, s.t); const u = N.normalize(s2);
        g.circle([0, 0], s.t, { fill: null, stroke: 'teal', width: 1.2, dashed: true, strokeAlpha: 0.6 * s.diskCmpA });
        g.vector(s2, N.add(s2, N.scale(u, 0.5)), { color: 'hud', width: 1.6, head: 8, alpha: 0.9 * s.diskCmpA, label: 'one normal', labelOffset: 12 });
        g.point(s2, { r: 4, color: 'teal', alpha: s.diskCmpA });
      }
      // eigenvalue lift (step 10)
      if (s.lift >= 0) {
        const e = N.eigSym(H); const lam = s.lift; const cc = 0.5;
        const Hl = [[H[0][0] + lam, H[0][1]], [H[1][0], H[1][1] + lam]];
        g.ellipse(ellipseM(H, cc), c, { stroke: 'gold', width: 1.4, dashed: true, strokeAlpha: 0.7, segments: 96 });
        g.ellipse(ellipseM(Hl, cc), c, { fill: 'gold', alpha: 0.12, stroke: 'gold', width: 2, segments: 96 });
        for (let k = 0; k < 2; k++) {
          const q = e.vectors[k], a = Math.sqrt(2 * cc / (e.values[k] + lam));
          g.vector(c, N.add(c, N.scale(q, a)), { color: 'gold', width: 2.6, label: 'λ' + (k ? '₂' : '₁') + ' = ' + LA.fmt(e.values[k]), labelOffset: k ? -18 : 16 });
          g.label('axis ' + LA.fmt(a), N.add(c, N.scale(q, a + 0.3)), { color: 'gold', size: 12 });
        }
      }
      // solutions
      if (s.showBoth) {
        const s1 = solveBall(H, c, 1, s.t), s2 = solveBall(H, c, 2, s.t);
        const far = q => N.dist(q, c) > 0.25;            // no labels once a solution has reached w*
        g.point(s1, { r: 6, color: 'mag', label: !far(s1) ? undefined : s.readouts ? 'lasso: w₂ = ' + LA.fmt.fixed(s1[1], 2) : 'lasso', dy: 16, dx: 14 });
        g.point(s2, { r: 6, color: 'mag', label: !far(s2) ? undefined : s.readouts ? 'ridge: w₂ = ' + LA.fmt.fixed(s2[1], 2) : 'ridge', dy: -14, dx: 16 });
        if (s.readouts) { g.dim(s2, [s2[0], 0], { label: LA.fmt.fixed(s2[1], 2), offset: 16, color: 'hud' }); g.line(s2, [s2[0], 0], { color: 'residual', dashed: true, width: 1.2 }); }
      } else if (s.showSol) {
        const lab = Math.abs(s.p - 1) < 1e-9 && s.mode === 'ball' && s.showBall ? 'w₂ = ' + LA.fmt.fixed(sol[1], 2) : 'w';
        g.point(sol, { r: 6, color: 'mag', label: lab, dy: -14, dx: 14 });
        if (s.glow) g.marker.ring(sol, { r: 14, color: 'mag' });
      }
      if (s.showGuess) {
        g.line(c, s.guess, { color: 'w', dashed: true, width: 1.4, alpha: 0.7 });
        g.point(s.guess, { r: 6.5, color: 'w', label: 'your w', dy: 14, dx: 12 });
      }
      if (s.crossA > 0.5) { g.label('“they do the same thing”', [-0.35, 2.75], { color: 'muted', size: 15, anchor: 'left', dx: 24 }); g.marker.cross([-0.35, 2.75], { size: 22, dx: 8 }); }
      g.point(c, { r: 5.5, color: 'gold', label: 'w*', dy: -12, dx: s.showBoth ? -14 : 10 });
      // handles
      g.handle('wstar', c, (p, st) => { st.wstar = [LA.clamp(p[0], -2.5, 3.2), LA.clamp(p[1], -2.5, 3.2)]; }, { color: 'gold' });
      if (s.mode === 'ball' && (s.showBall || s.showBoth) && !s.noBudget) g.handle('budget', [(s.wstar[0] >= 0 ? 1 : -1) * s.t, 0], (p, st) => { st.t = LA.clamp(Math.abs(p[0]), 0.02, 3.5); }, { color: 'teal' });
    },

    math(s, tex) {
      const H = getH(s), c = s.wstar; const m = s.mathMode;
      const wsT = tex.sym('w', { sup: '*', color: 'gold' });
      const Hm = tex.mat(H, { digits: 2 });
      if (m === 'loss') return tex.lines(tex.eq(tex.sym('L'), tex.paren(tex.sym('w')), tex.op('='), tex.text('½'), tex.T(tex.paren(tex.sym('w') + tex.op('−') + wsT)), tex.sym('H'), tex.paren(tex.sym('w') + tex.op('−') + wsT)), tex.eq(tex.sym('H'), tex.op('='), Hm, tex.op('·'), wsT, tex.op('='), tex.vec(c, { color: 'gold' })));
      if (m === 'budget') return tex.lines(tex.eq(tex.text('min'), tex.sym('L'), tex.paren(tex.sym('w')), tex.text('subject to'), tex.norm(tex.sym('w')), '<sub>' + (Math.abs(s.p - 1) < 1e-9 ? '1' : '2') + '</sub>', tex.op('≤'), tex.sym('t'), tex.op('='), tex.num(s.t, { color: 'teal' })), Math.abs(s.p - 1) < 1e-9 ? tex.eq(tex.sym('w', { sub: '2', color: 'mag' }), tex.op('='), tex.num(currentSol(s)[1], { color: 'mag' })) : tex.eq(tex.sym('w', { color: 'mag' }), tex.op('='), tex.vec(currentSol(s), { color: 'mag' })));
      if (m === 'tangent') return tex.lines(tex.eq(tex.sym('∇L', { color: 'v' }), tex.op('∥'), tex.sym('w', { color: 'w' }), tex.op('⇔'), tex.sym('H'), tex.paren(tex.sym('w') + tex.op('−') + wsT), tex.op('+'), tex.sym('λ'), tex.sym('w'), tex.op('='), tex.num(0)), tex.eq(tex.sym('∇L', { color: 'v' }), tex.op('='), tex.vec(grad(H, c, currentSol(s)), { color: 'v' }), tex.op('·'), tex.sym('w', { color: 'w' }), tex.op('='), tex.vec(currentSol(s), { color: 'w' })));
      if (m === 'ridge') return tex.lines(tex.eq(tex.sym('w', { sub: 'ridge', color: 'mag' }), tex.op('='), tex.inv(tex.paren(tex.sym('H') + tex.op('+') + tex.sym('λ') + tex.sym('I'))), tex.sym('H'), wsT), tex.eq(tex.sym('λ'), tex.op('='), tex.num(s.lambda, { color: 'teal' }), tex.op('·'), tex.sym('w', { color: 'mag' }), tex.op('='), tex.vec(ridgeSol(H, c, s.lambda), { color: 'mag' })));
      if (m === 'fan') { const ng = N.scale(grad(H, c, [s.fanAt, 0]), -1); return tex.lines(tex.eq(tex.text('corner fan:'), tex.abs(tex.sym('b')), tex.op('≤'), tex.sym('a')), tex.eq(tex.op('−'), tex.sym('∇L', { color: 'v' }), tex.op('='), tex.vec(ng, { color: 'v' }), tex.text(Math.abs(ng[1]) <= ng[0] + 1e-9 ? 'inside the fan' : 'outside the fan', { color: Math.abs(ng[1]) <= ng[0] + 1e-9 ? 'ok' : 'bad' }))); }
      if (m === 'paths') { const kt = kinkT(H, c); return tex.lines(tex.eq(tex.text('lasso:'), tex.sym('w', { sub: '2' }), tex.op('='), tex.num(0), tex.text('for'), tex.sym('t'), tex.op('≤'), tex.num(kt, { color: 'b' })), tex.eq(tex.text('ridge:'), tex.sym('w', { sub: '2' }), tex.op('>'), tex.num(0), tex.text('for all'), tex.sym('t'), tex.op('>'), tex.num(0)), tex.eq(tex.sym('t'), tex.op('='), tex.num(s.t, { color: 'teal' }))); }
      if (m === 'miscon') { const s1 = solveBall(H, c, 1, s.t), s2 = solveBall(H, c, 2, s.t); return tex.lines(tex.eq(tex.text('ridge'), tex.sym('w', { sub: '2' }), tex.op('='), tex.num(s2[1], { color: 'mag' }), tex.op('≠'), tex.text('lasso'), tex.sym('w', { sub: '2' }), tex.op('='), tex.num(s1[1], { color: 'mag' }))); }
      if (m === 'corner') return tex.lines(tex.eq(tex.text('corner contact only for'), tex.sym('t'), tex.op('≤'), tex.num(kinkT(H, c), { color: 'b' })), tex.eq(wsT, tex.op('='), tex.vec(c, { color: 'gold' })));
      if (m === 'lift') { const e = N.eigSym(H); const lam = Math.max(0, s.lift); return tex.lines(tex.eq(tex.text('axis length'), tex.frac('1', tex.sqrt(tex.sym('λ', { sub: 'i' }))), tex.op('→'), tex.frac('1', tex.sqrt(tex.sym('λ', { sub: 'i' }) + tex.op('+') + tex.sym('λ')))), tex.eq(tex.sym('λ'), tex.op('='), tex.num(lam, { color: 'teal' }), tex.op('·'), tex.sym('λ', { sub: '1' }), tex.op('+'), tex.sym('λ'), tex.op('='), tex.num(e.values[0] + lam, { color: 'gold' }), tex.op('·'), tex.sym('λ', { sub: '2' }), tex.op('+'), tex.sym('λ'), tex.op('='), tex.num(e.values[1] + lam, { color: 'gold' }))); }
      if (m === 'wd') return tex.lines(tex.eq(tex.sym('w'), tex.op('←'), tex.paren(tex.num(1) + tex.op('−') + tex.sym('η') + tex.sym('λ')), tex.sym('w'), tex.op('−'), tex.sym('η'), tex.sym('∇L', { color: 'v' })), tex.eq(tex.text('after'), tex.num(s.wdK), tex.text('steps:'), tex.norm(tex.sym('w')), tex.op('×'), tex.num(Math.pow(0.85, s.wdK), { color: 'w' })));
      if (m === 'guess') return tex.lines(tex.eq(tex.sym('w', { sub: 'ridge', color: 'gold' }), tex.op('='), tex.inv(tex.paren(tex.sym('I') + tex.op('+') + tex.num(1) + tex.sym('I'))), tex.sym('I'), wsT, tex.op('·'), tex.text('yours:'), tex.vec(s.guess, { color: 'w' })));
      return '';
    },

    steps: [
      { title: 'The unconstrained minimum', duration: 3,
        narration: 'A regression with two weights has a loss bowl (p5-03) whose bottom is ' + ws + ' = (2, 1), drawn gold. Its contours are the gray ellipses, shaped by H = [[1, 0.5],[0.5, 1]]. With no constraint, training follows the faint orange path, lands on w* and stops there.',
        setup(s) { s.mode = 'ball'; s.p = 2; s.t = 0; s.lambda = 0; s.wstar = WS0.slice(); s.showBall = false; s.showSol = false; s.showPaths = false; s.showBoth = false; s.readouts = false; s.tangentA = 0; s.gradA = 0; s.fanA = 0; s.diskCmpA = 0; s.crossA = 0; s.lift = -1; s.mlTie = false; s.recap = false; s.glow = false; s.showGuess = false; s.mathMode = 'loss'; s.gd = 0; },
        at(s, t) { s.gd = seg(t, 0.1, 0.9, 'linear'); } },

      { title: 'A budget: the L2 disk', duration: 4,
        narration: 'Now limit the size of w: ‖w‖₂ ≤ t, a <b>norm ball</b> from p1-05 — here the teal disk. Grow t from zero. The smallest loss you can afford is where the disk first touches a contour, the gold ellipse: the magenta point. That contact point is the solution under <b>regularization</b>.',
        setup(s) { s.gd = 0; s.p = 2; s.showBall = true; s.showSol = true; s.mathMode = 'budget'; s.t = 0; s.glow = false; },
        at(s, t) { s.t = 1.1 * seg(t, 0.05, 0.85); s.glow = t > 0.9; } },

      { title: 'A smooth touch', duration: 3,
        narration: 'At the touch the disk and the gold ellipse share a tangent, the dashed gray line. Their normals line up: the yellow loss gradient <span class="c-v">∇L</span> points one way along it, the purple <span class="c-w">w</span> itself the other. Written out, that is H(w − w*) + λw = 0 for some λ.',
        setup(s) { s.t = 1.1; s.glow = false; s.mathMode = 'tangent'; s.tangentA = 0; s.gradA = 0; },
        at(s, t) { s.tangentA = seg(t, 0, 0.4); s.gradA = seg(t, 0.35, 0.8); } },

      { title: 'Same thing, penalty form: ridge', duration: 4,
        narration: 'Solve that equation and you get <b>ridge</b> from p4-12: w = (H + λI)⁻¹Hw*. The budget t and the penalty λ are two dials for one knob. Slide λ up: the magenta solution slides smoothly from w* toward the origin along the orange <b>solution path</b>, never touching an axis.',
        setup(s) { s.tangentA = 0; s.gradA = 0; s.mode = 'ridge'; s.lambda = 0; s.mathMode = 'ridge'; },
        at(s, t) { s.lambda = 6 * seg(t, 0.05, 0.95); } },

      { title: 'A different budget: the L1 diamond', duration: 4,
        narration: 'Swap the disk for the teal diamond ‖w‖₁ ≤ t, the L1 ball. Grow it. The first contact is at the corner (t, 0): the second weight is exactly zero. Not small — zero. This is <b>sparsity</b>, and the diamond’s name for it is <b>lasso</b>.',
        setup(s) { s.mode = 'ball'; s.p = 1; s.t = 0; s.lambda = 0; s.mathMode = 'budget'; },
        at(s, t) { s.t = seg(t, 0.05, 0.85); } },

      { title: 'Why corners win', duration: 4,
        narration: 'At a corner the diamond has no single tangent; a whole fan of gray directions counts as outward, bounded by the two green rays. Any ellipse whose downhill arrow −<span class="c-v">∇L</span> falls inside that fan touches the corner. The disk’s edge has one normal per point, so contact lands on an axis only by accident.',
        setup(s) { s.t = 1; s.fanAt = 1; s.mathMode = 'fan'; s.fanA = 0; s.diskCmpA = 0; },
        at(s, t) { s.fanA = seg(t, 0, 0.45); s.diskCmpA = seg(t, 0.55, 0.95); } },

      { title: 'The two paths', duration: 4,
        narration: 'Sweep the budget for both balls. The solid orange lasso path runs along the w₁ axis up to (1, 0), then turns toward w*: a kink where w₂ switches on. The dashed orange ridge path is a smooth curve that never reaches an axis. Both end at w* once the budget is large.',
        setup(s) { s.fanA = 0; s.diskCmpA = 0; s.showBall = false; s.showSol = false; s.showPaths = true; s.showBoth = true; s.mathMode = 'paths'; s.t = 0; },
        at(s, t) { s.t = 3 * seg(t, 0.05, 0.95, 'linear'); } },

      { title: '✕ “They both just shrink the weights”', duration: 3, misconception: true,
        narration: 'The claim: “L1 and L2 do the same thing, a little more or less.” Look at the paths. At budget t = 1 ridge gives w₂ = 0.59; lasso gives 0.00. One shrinks everything a bit; the other switches a weight off. Different shapes, different answers.',
        setup(s) { s.t = 1; s.readouts = true; s.mathMode = 'miscon'; s.crossA = 0; },
        at(s, t) { s.crossA = seg(t, 0.3, 0.6); } },

      { title: 'When the corner loses', duration: 4, interactive: true,
        narration: 'Drag ' + ws + ' to (2, 1.8). Now the downhill arrow at the corner falls outside the fan for all but tiny budgets, so the kink moves to t = 0.2: both weights switch on almost at once. Sparsity is not guaranteed; it depends on where the data puts w* and how the ellipses are tilted.',
        setup(s) { s.readouts = false; s.crossA = 0; s.mathMode = 'corner'; s._ws0 = LA.clone(s.wstar); s._ws1 = [2, 1.8]; s.t = 1; },
        at(s, t) { s.wstar = lerpV(s._ws0, s._ws1, seg(t, 0.1, 0.7)); s.fanAt = kinkT(getH(s), s.wstar); s.fanA = seg(t, 0.6, 0.9); } },

      { title: 'Ridge as an eigenvalue lift', duration: 4,
        narration: 'From p4-12: ridge adds λ to every eigenvalue of H. Watch the gold ellipse. The gold axis along the small eigenvalue — the direction the data barely constrains — shrinks fastest: its length goes from 1/√λᵢ to 1/√(λᵢ + λ), here 1.41 → 0.53. Ridge shrinks most where the data knows least.',
        setup(s) { s.wstar = WS0.slice(); s.fanA = 0; s.showPaths = false; s.showBoth = false; s.mathMode = 'lift'; s.lift = 0; },
        at(s, t) { s.lift = 3 * seg(t, 0.1, 0.9); } },

      { title: 'Why ML cares: weight decay and sparse models', duration: 4, mlTie: true,
        narration: 'In code, L2 is <b>weight decay</b>: each step multiplies the purple w by (1 − ηλ) before the gradient step. L1 switches features off — model selection by geometry. In n-D the diamond has 2n corners on the axes, so zeros are even more common than this picture suggests. The picture stops at 3. The formula does not.',
        setup(s) { s.lift = -1; s.mlTie = true; s.mathMode = 'wd'; s.mlA = 0; s.wdK = 0; s._cam0 = LA.clone(s.cam); s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.0, 1.3], halfH: 2.4 }); },
        at(s, t) { s.mlA = seg(t, 0.1, 0.5); s.wdK = Math.floor(seg(t, 0.35, 1, 'linear') * 8); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Regularization is a tug-of-war between the loss bowl and a norm ball. Contact at a smooth edge shrinks smoothly (ridge). Contact at a corner on an axis sets a weight to zero (lasso). The paths show the difference: curved versus kinked. Elastic net mixes the two balls.',
        cards: ['<b>Solution = first touch</b> of the norm ball and a loss contour.', '<b>Corners on axes ⇒ exact zeros.</b> The diamond touches at (t, 0) whenever −∇L falls in the fan.', '<b>Ridge path curves, lasso path kinks.</b> w = (H + λI)⁻¹Hw* vs. a weight switching on.'],
        setup(s) { s.mlTie = false; s.recap = true; s.mathMode = 'none'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.15, 1.05], halfH: 2.6 }); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag <span class="c-gold">w*</span> or the teal budget handle on the w₁ axis; change the norm p (1 = diamond, 2 = disk) and the tilt of the loss. The magenta point is the constrained solution; the orange curves are the lasso (solid) and ridge (dashed) paths.</p>',
      setup(s) { s.mode = 'ball'; s.p = 1; s.t = 1; s.showBall = true; s.showSol = true; s.showPaths = true; s.showBoth = false; s.mathMode = 'budget'; },
      controls: [
        { type: 'slider', bind: 'p', min: 0.5, max: 3, step: 0.5, label: 'Norm p (1 = L1 diamond, 2 = L2 disk)', color: 'teal' },
        { type: 'slider', bind: 't', min: 0.02, max: 3, step: 0.02, label: 'Budget t', color: 'teal' },
        { type: 'slider', bind: 'rho', min: -0.9, max: 0.9, step: 0.1, label: 'Tilt of the loss (H₁₂)' },
        { type: 'toggle', bind: 'showPaths', label: 'Solution paths' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Default', values: { rho: 0.5, wstar: [2, 1], p: 1, t: 1 } },
          { label: 'Corner loses', values: { rho: 0.5, wstar: [2, 1.8], p: 1, t: 1 } },
          { label: 'Round contours (H = I)', values: { rho: 0, wstar: [2, 1], p: 2, t: 1 } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag ' + ws + ' to a place where the L1 solution at budget t = 1 has w₂ = 0 exactly.',
        setup(s) { s.rho = 0.5; s.mode = 'ball'; s.p = 1; s.t = 1; s.noBudget = true; s.wstar = [2, 1.8]; s.showBall = true; s.showSol = true; s.showPaths = false; s.mathMode = 'budget'; },   // budget locked at t = 1
        judge(s) { const sol = solveBall(getH(s), s.wstar, 1, 1); const ok = Math.abs(sol[1]) < 0.02 && Math.abs(sol[0]) > 0.2; return { ok, progress: Math.max(0, 1 - Math.abs(sol[1]) / 0.6) }; },
        hint: 'Bring w* down toward the w₁ axis so the downhill arrow at the corner falls inside the fan.',
        solution(s) { s.wstar = [2.5, 0.6]; } },
      { kind: 'manipulate',
        prompt: 'Slide the teal budget handle to the largest t at which the L1 solution still has w₂ = 0 (the kink).',
        setup(s) { s.rho = 0.5; s.mode = 'ball'; s.p = 1; s.noBudget = false; s.wstar = [2, 1]; s.t = 0.3; s.showBall = true; s.showSol = true; s.showPaths = false; s.mathMode = 'budget'; },   // no paths: the kink marker would show the answer
        judge(s) { const kt = kinkT(getH(s), s.wstar); return LA.judge.scalar(s.t, kt, LA.TOL.pos); },
        hint: 'Watch w₂ in the readout: it switches on just past the kink. Stop right there.',
        solution(s) { s.t = kinkT(getH(s), s.wstar); } },
      { kind: 'entry',
        prompt: 'Type the ridge solution for λ = 1 when H = I and w* = (2, 1). Your point is drawn purple, joined to ' + ws + ' by a dashed line.',
        bind: 'guess', answer: [1, 0.5], tol: LA.TOL.entry, label: 'w', colors: 'w',
        setup(s) { s.rho = 0; s.noBudget = false; s.wstar = [2, 1]; s.mode = 'ridge'; s.lambda = 1; s.showBall = true; s.showSol = false; s.showPaths = false; s.showGuess = true; s.guess = [0, 0]; s.mathMode = 'guess'; },
        hint: '(I + I)⁻¹ I w* halves w*.' },
    ],
  });

  /* ---------- ML tie-in scene ---------- */
  function renderML(g, s) {
    const W = g.size.w, Hh = g.size.h;
    const d = stripData();
    g.ghostGrid({ range: 6 });                       // grid first, so the strips sit on top of it
    g.axes({ names: ['w₁', 'w₂'], labels: false });
    const x0 = 28, w = Math.min(W - 56, 560), h = 22, cw = w / 12;
    const y1 = 58, y2 = 142;
    g.hud('12 weights of a synthetic model (seed 504) — before', x0, y1 - 16, { size: 12.5 });
    g.strip(d.before, { x: x0, y: y1, w, h, range: [-2.6, 2.6], ramp: divRamp });
    d.before.forEach((v, i) => g.label(LA.fmt.fixed(v, 2), [x0 + (i + 0.5) * cw, y1 + h + 10], { screen: true, size: 10, color: 'hud', font: 'mono', bg: false }));
    if (s.mlA > 0.2) {
      const nz = d.after.filter(x => x === 0).length;
      g.hud('after lasso — ' + nz + ' weights are exactly 0', x0, y2 - 16, { size: 12.5 });
      g.strip(d.after, { x: x0, y: y2, w, h, range: [-2.6, 2.6], ramp: divRamp });
      d.after.forEach((v, i) => g.label(v === 0 ? '0' : LA.fmt.fixed(v, 2), [x0 + (i + 0.5) * cw, y2 + h + 10], { screen: true, size: 10, color: v === 0 ? 'mag' : 'hud', font: 'mono', bold: v === 0, bg: false }));
    }
    g.hud('The picture stops at 3. The formula does not.', x0, Hh - 34, { size: 14, color: 'text' });
    g.hud('see p6-08', W - 22, Hh - 34, { size: 12, color: 'muted', anchor: 'right' });
    // weight decay on one arrow
    const w0 = [2.2, 1.3], f = 0.85;
    for (let k = 0; k <= s.wdK; k++) {
      const v = N.scale(w0, Math.pow(f, k)); const last = k === s.wdK;
      g.vector(v, { color: 'w', width: last ? 3 : 1.4, alpha: last ? 1 : 0.3, label: last ? 'w' : undefined, head: last ? 12 : 7 });
    }
    g.label('× (1 − ηλ) each step, ' + s.wdK + ' step' + (s.wdK === 1 ? '' : 's'), [0.1, 1.9], { color: 'w', anchor: 'left', size: 13 });
  }

  /* ---------- recap: three mini-cards ---------- */
  function renderRecap(g, s) {
    const H = getH(s), c = WS0;
    const cards = [
      { title: 'disk: smooth touch', cam: { center: [1.0, 0.9], halfH: 1.7 }, draw(g2) { for (const lv of LEVELS) g2.ellipse(ellipseM(H, lv), c, { stroke: 'residual', width: 1, strokeAlpha: 0.5 }); g2.circle([0, 0], 1.1, { fill: 'teal', alpha: 0.2, stroke: 'teal' }); const so = solveBall(H, c, 2, 1.1); g2.ellipse(ellipseM(H, loss(H, c, so)), c, { stroke: 'gold', width: 1.6 }); g2.point(so, { r: 5, color: 'mag' }); g2.point(c, { r: 4, color: 'gold' }); } },
      { title: 'diamond: corner ⇒ w₂ = 0', cam: { center: [1.0, 0.9], halfH: 1.7 }, draw(g2) { for (const lv of LEVELS) g2.ellipse(ellipseM(H, lv), c, { stroke: 'residual', width: 1, strokeAlpha: 0.5 }); g2.polygon(ballPts(1, 1, 4), { fill: 'teal', alpha: 0.2, stroke: 'teal' }); const so = solveBall(H, c, 1, 1); g2.ellipse(ellipseM(H, loss(H, c, so)), c, { stroke: 'gold', width: 1.6 }); g2.point(so, { r: 5, color: 'mag', label: 'w₂ = 0', dy: 14 }); g2.point(c, { r: 4, color: 'gold' }); } },
      { title: 'curved vs kinked paths', cam: { center: [1.0, 0.9], halfH: 1.7 }, draw(g2) { for (const lv of LEVELS) g2.ellipse(ellipseM(H, lv), c, { stroke: 'residual', width: 1, strokeAlpha: 0.5 }); g2.polyline(pathFor(H, c, 1).map(q => q.w), { color: 'b', width: 2.2 }); g2.polyline(pathFor(H, c, 2).map(q => q.w), { color: 'b', width: 2.2, dashed: true }); g2.point([kinkT(H, c), 0], { r: 4, color: 'b' }); g2.point(c, { r: 4, color: 'gold', label: 'w*', dy: -12 }); } },
    ];
    const W = g.size.w, Hh = g.size.h, pad = 10, cw = (W - pad * 4) / 3, ch = Math.min(Hh * 0.6, cw * 1.25), y = (Hh - ch) / 2;
    cards.forEach((cd, i) => {
      const r = { x: pad + i * (cw + pad), y, w: cw, h: ch };
      g.viewport(r, { view: '2d', cam: Object.assign(LA.cam.defaults2d(), cd.cam), bg: 'panel' }, (g2, rp) => { g2.ghostGrid({ range: 5 }); g2.axes({ labels: false }); cd.draw(g2); g2.label(cd.title, [rp.x + rp.w / 2, rp.y + 16], { screen: true, size: 13, color: 'text', bold: true }); });
    });
  }
})();
