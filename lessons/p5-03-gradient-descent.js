/* p5-03 — Gradient descent on an elongated vs round bowl (capstone). L(w) = ½wᵀHw, ∇L = Hw, w ← w − η∇L.
   Each eigen-coordinate shrinks by 1 − ηλᵢ: the steep one sets the stability limit 2/λmax, the gentle one the crawl.
   Paths are precomputed by deterministic loops (setup, or a pure function of t when η / λ sweep) and revealed by t. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, seg = LA.seg, fmt = LA.fmt;
  const PI = Math.PI;
  const PHI = -PI / 4;                                   // angle of q₁ (the steep axis) for the rotated bowl
  const LAM = [10, 1], W0 = [2.5, 1.5], ETA = 0.15;
  const LV = [0.5, 1, 2, 4, 8, 16];
  const CB = { x0: -4.2, x1: 4.2, y0: -4.2, y1: 4.2 };
  const ZS = 0.25, CAP = 10, RMAX = 3.2;
  const BETA_M = 0.5;                                    // momentum used in step 11 (see report: 0.8 is slower than plain GD here)

  const Hof = (lam, phi) => { const c = Math.cos(phi), s = Math.sin(phi), o = (lam[0] - lam[1]) * c * s; return [[lam[0] * c * c + lam[1] * s * s, o], [o, lam[0] * s * s + lam[1] * c * c]]; };
  const qs = phi => [[Math.cos(phi), Math.sin(phi)], [-Math.sin(phi), Math.cos(phi)]];
  const Lq = (H, w) => 0.5 * (H[0][0] * w[0] * w[0] + 2 * H[0][1] * w[0] * w[1] + H[1][1] * w[1] * w[1]);
  const grad = (H, w) => [H[0][0] * w[0] + H[0][1] * w[1], H[0][1] * w[0] + H[1][1] * w[1]];
  const rot = (a, p) => [Math.cos(a) * p[0] - Math.sin(a) * p[1], Math.sin(a) * p[0] + Math.cos(a) * p[1]];
  /** Deterministic GD / heavy-ball loop: v ← βv − η∇L, w ← w + v (β = 0 is plain GD). Stops once it has flown far out. */
  function gd(H, w0, eta, beta, n) {
    const pts = [[w0[0], w0[1]]]; let w = [w0[0], w0[1]], v = [0, 0];
    for (let k = 0; k < n; k++) {
      const g = grad(H, w); v = [beta * v[0] - eta * g[0], beta * v[1] - eta * g[1]]; w = [w[0] + v[0], w[1] + v[1]];
      pts.push([w[0], w[1]]);
      if (!(Math.abs(w[0]) + Math.abs(w[1]) < 1e6)) break;
    }
    return pts;
  }
  const cam2 = () => Object.assign(LA.cam.defaults2d(), { center: [0.55, 0.45], halfH: 3.6 });
  const CAMZ = { center: [1.62, 1.45], halfH: 1.25 };   // close-up of the first four steps
  const cam3 = o => Object.assign(LA.cam.defaults3d(), { target: [0.3, 0.3, 0.5], dist: 13.5, yaw: -1.4, pitch: 0.62, fov: 36 }, o || {});
  // Engine gap: tex.reconcile mis-assigns tex.num values when a panel also holds a tex.mat/vec; plain colored spans
  // are part of the skeleton, so the panel just rebuilds.
  const numH = (tex, x, o) => { o = o || {}; return '<span class="tx-n ' + tex.colorClass(o.color) + '">' + (typeof x === 'number' ? fmt(x, o.digits) : x) + '</span>'; };
  function makeData() { return LA.rng.gaussianCloud(503, 40, { mean: [0, 0], cov: [[1, 0], [0, 9]] }).map(p => [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100]); }
  function makeML() {
    const rand = LA.rng(503), B = [];
    for (let i = 0; i < 6; i++) { const r = []; for (let j = 0; j < 6; j++) r.push(LA.rng.gauss(rand)); B.push(r); }
    const A = N.scaleM(N.matMul(N.transpose(B), B), 1 / 6).map(r => r.map(x => Math.round(x * 100) / 100));
    const strip = []; for (let i = 0; i < 12; i++) strip.push(Math.round(LA.clamp(LA.rng.gauss(rand) * 0.5, -1, 1) * 100) / 100);
    return { heat6: A, strip };
  }
  /** a dashed 3D segment as short solid pieces (canvas dashes on 3D segments blanked some headless frames) */
  function dash3(g, a, b, o) { const n = 12; for (let i = 0; i < n; i += 2) g.line(lerpV(a, b, i / n), lerpV(a, b, (i + 1) / n), o); }

  LA.registerLesson({
    id: 'p5-03-gradient-descent',
    part: 5,
    title: 'Gradient descent on an elongated vs round bowl',
    subtitle: 'Each eigen-coordinate shrinks by 1 − ηλ: the steep axis caps the step, the gentle one sets the crawl.',
    minutes: 14,
    prereqs: ['p5-01-gradient', 'p5-02-hessian-curvature', 'p4-06-symmetric-quadratic-forms', 'p4-12-pseudoinverse-conditioning', 'p4-01-change-of-basis'],
    tags: ['capstone'],
    badge: { drawnIn: '2D weight space (two weights) + 3D loss surface', realSize: 'ResNet-50 has 2.5×10⁷ weights — a bowl with that many axes', transfers: 'w ← w − η∇L, the per-eigendirection factor 1 − ηλᵢ, η < 2/λmax, κ = λmax/λmin controls step count — exact for any quadratic in any n, and the local truth near any minimum' },
    cantShow: ['Non-convex landscapes with many minima and saddles; we draw one bowl.', 'Mini-batch noise on the gradient (stochastic gradient descent).', 'The other n − 2 eigendirections, each shrinking by its own 1 − ηλᵢ.'],
    terms: [
      { term: 'learning rate', def: 'The step-length multiplier η in w ← w − η∇L.' },
      { term: 'gradient descent', def: 'Repeating the step w ← w − η∇L.' },
      { term: 'zig-zag', def: 'Alternating overshoot back and forth across a steep direction.' },
      { term: 'condition number', def: 'κ = λmax/λmin of the Hessian: how elongated the bowl is.' },
      { term: 'stability limit', def: 'η must stay below 2/λmax, or the steep coordinate grows instead of shrinking.' },
      { term: 'standardize', def: 'Rescale each feature to unit spread, which rounds the loss bowl.' },
      { term: 'momentum', def: 'Carry a fraction β of the previous step into the next one.' },
    ],
    atlas: ['optimization', 'normalization', 'linear-regression'],
    thumbnail: 2,

    state() {
      const ml = makeML();
      return {
        view: '2d', cam: cam2(),
        lam: LAM.slice(), phi: PHI, w0: W0.slice(), eta: ETA, beta: 0, steps: 12,
        path: gd(Hof(LAM, PHI), W0, ETA, 0, 12), pathM: [], k: 0,
        eigView: 0, showEig: true, eigGrow: 1, showStart: true, gradShow: 0, stepShow: 0, decomp: false,
        etaAxis: false, factorBars: false, aBars: false, showLine: true, split: null, stdU: 0,
        data: makeData(), heat6: ml.heat6, strip: ml.strip, ml: 0, recap: false, panel: 'bowl', hudL: false,
        preset: 'rotated',
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g, s); return; }
      if (s.ml > 0) { renderML(g, s); return; }
      if (s.split === 'mis') { renderSplitMis(g, s); return; }
      if (s.split === 'std') { renderSplitStd(g, s); return; }
      const live = ctx.mode !== 'learn';
      const H = Hof(s.lam, s.phi);
      const path = live ? gd(H, s.w0, s.eta, s.beta, Math.round(s.steps)) : s.path;
      const k = live ? path.length - 1 : s.k;
      if (s.view === '3d') render3d(g, s, ctx, H, path, k); else render2d(g, s, ctx, H, path, k, live);
    },

    math(s, tex) {
      tex = Object.assign({}, tex, { num: (x, o) => numH(tex, x, o) });
      const H = Hof(s.lam, s.phi), T = str => tex.text(str, { html: true });
      const lmax = Math.max(s.lam[0], s.lam[1]), lmin = Math.min(s.lam[0], s.lam[1]);
      const Hm = tex.mat(H.map(r => r.map(x => Math.round(x * 100) / 100)));
      const G = tex.sym('∇L', { color: 'v' });
      const kk = Math.min(s.path.length - 1, Math.floor(s.k + 1e-9)), wk = s.path[Math.max(0, kk)] || s.w0;
      switch (s.panel) {
        case 'bowl': return tex.lines(tex.eq(T('L(w) = ½ w<sup>ᵀ</sup>H w'), tex.op('·'), tex.sym('H'), tex.op('='), Hm), tex.eq(T('λ'), tex.op('='), tex.num(s.lam[0], { color: 'gold' }), tex.op(','), tex.num(s.lam[1], { color: 'gold' }), tex.op('·'), T('κ = λ<sub>max</sub>/λ<sub>min</sub>'), tex.op('='), tex.num(lmax / lmin)));
        case 'step': { const g0 = grad(H, s.w0), w1 = [s.w0[0] - ETA * g0[0], s.w0[1] - ETA * g0[1]]; return tex.lines(tex.eq(tex.sym('w', { color: 'b' }), tex.op('←'), tex.sym('w', { color: 'b' }), tex.op('−'), tex.sym('η'), G, tex.op('·'), G, tex.op('='), T('Hw'), tex.op('·'), tex.sym('η'), tex.op('='), tex.num(ETA)), tex.eq(G, tex.op('='), tex.vec(g0, { color: 'v' }), tex.op('→'), tex.sym('w', { sub: '1', color: 'b' }), tex.op('='), tex.vec(w1, { color: 'b' }))); }
        case 'many': return tex.block(tex.eq(tex.sym('k'), tex.op('='), tex.num(kk, { color: 'b' }), tex.op('·'), T('L(w<sub>k</sub>)'), tex.op('='), tex.num(Lq(H, wk), { digits: 3 }), tex.op('·'), T('‖w<sub>k</sub>‖'), tex.op('='), tex.num(N.norm(wk))));
        case 'why': return tex.block(tex.eq(T('step = (−ηλ<sub>1</sub>a<sub>1</sub>)<span class="c-gold">q₁</span> + (−ηλ<sub>2</sub>a<sub>2</sub>)<span class="c-gold">q₂</span>')), tex.eq(T('−ηλ<sub>1</sub> = −1.5'), tex.op('·'), T('−ηλ<sub>2</sub> = −0.15')));
        case 'eig': { const Q = qs(s.phi), a1 = N.dot(wk, Q[0]), a2 = N.dot(wk, Q[1]); return tex.lines(tex.eq(T('a<sub>i</sub> ← (1 − ηλ<sub>i</sub>) a<sub>i</sub>'), tex.op('·'), T('1 − ηλ<sub>1</sub>'), tex.op('='), tex.num(1 - ETA * s.lam[0], { color: 'gold' }), tex.op('·'), T('1 − ηλ<sub>2</sub>'), tex.op('='), tex.num(1 - ETA * s.lam[1], { color: 'gold' })), tex.eq(T('a<sub>1</sub>'), tex.op('='), tex.num(a1, { digits: 3 }), tex.op(','), T('a<sub>2</sub>'), tex.op('='), tex.num(a2, { digits: 3 }), tex.text('at k = ' + kk))); }
        case 'eta': return tex.lines(tex.eq(tex.text('stable'), tex.op('⇔'), T('|1 − ηλ<sub>i</sub>| < 1'), tex.op('⇔'), T('η < 2/λ<sub>max</sub>'), tex.op('='), tex.num(2 / lmax, { color: 'bad' })), tex.eq(tex.sym('η'), tex.op('='), tex.num(s.eta, { color: s.eta > 2 / lmax ? 'bad' : 'b' }), tex.op('·'), T('|1 − ηλ<sub>1</sub>|'), tex.op('='), tex.num(Math.abs(1 - s.eta * s.lam[0]), { color: Math.abs(1 - s.eta * s.lam[0]) > 1 ? 'bad' : '' }), tex.op('·'), T('|1 − ηλ<sub>2</sub>|'), tex.op('='), tex.num(Math.abs(1 - s.eta * s.lam[1]))));
        case 'mis': return tex.block(tex.eq(T('0.99<sup>40</sup>'), tex.op('≈'), tex.num(0.67, { color: 'bad' }), tex.op('·'), T('0.85<sup>40</sup>'), tex.op('≈'), tex.num(0.0015, { color: 'ok', digits: 4 })));
        case 'kappa': { const kap = lmax / lmin; return tex.lines(tex.eq(T('κ = λ<sub>max</sub>/λ<sub>min</sub>'), tex.op('='), tex.num(kap), tex.op('·'), T('η = 2/(λ<sub>1</sub> + λ<sub>2</sub>)'), tex.op('='), tex.num(s.eta)), tex.eq(tex.text('best factor'), T('(κ − 1)/(κ + 1)'), tex.op('='), tex.num((kap - 1) / (kap + 1), { color: 'gold' }))); }
        case 'std': return tex.block(tex.eq(T('H = X<sup>ᵀ</sup>X/n'), tex.op('='), Hm, tex.text(s.stdU > 0.98 ? '= I after standardizing' : '(raw features)')));
        case 'mom': return tex.block(tex.eq(T('v ← βv − η'), G, tex.op('·'), T('w ← w + v'), tex.op('·'), T('β'), tex.op('='), tex.num(BETA_M, { color: 'w' }), tex.op(','), T('η'), tex.op('='), tex.num(ETA)));
        case 'ml': return tex.block(tex.eq(T('L(w) ≈ L(w*) + ½(w − w*)<sup>ᵀ</sup>H(w − w*)')));
        case 'none': return '';
        default: {
          const path = gd(H, s.w0, s.eta, s.beta, Math.round(s.steps)), wn = path[path.length - 1];
          return tex.lines(tex.eq(tex.sym('H'), tex.op('='), Hm, tex.op('·'), T('κ'), tex.op('='), tex.num(lmax / lmin), tex.op('·'), T('2/λ<sub>max</sub>'), tex.op('='), tex.num(2 / lmax, { color: 'bad' })),
            tex.eq(tex.sym('η'), tex.op('='), tex.num(s.eta, { color: s.eta > 2 / lmax ? 'bad' : 'b' }), tex.op('·'), tex.sym('β'), tex.op('='), tex.num(s.beta, { color: 'w' }), tex.op('·'), T('‖w<sub>' + (path.length - 1) + '</sub>‖'), tex.op('='), tex.num(N.norm(wn) > 1e4 ? Infinity : N.norm(wn), { digits: 3 })));
        }
      }
    },

    steps: [
      { title: 'The bowl we will roll down', duration: 3,
        narration: 'A loss over two weights: L(w) = ½wᵀHw with H = [[5.5, −4.5], [−4.5, 5.5]]. From p5-02 you can read its shape: gold axes at ±45°, curvature 10 across the valley, 1 along it. The <b>condition number</b> is 10 — a long, narrow bowl.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.lam = LAM.slice(); s.phi = PHI; s.w0 = W0.slice(); s.eta = ETA; s.beta = 0; s.path = gd(Hof(LAM, PHI), W0, ETA, 0, 12); s.k = 0; s.showStart = false; s.showEig = true; s.eigGrow = 0; s.gradShow = 0; s.stepShow = 0; s.decomp = false; s.eigView = 0; s.etaAxis = false; s.factorBars = false; s.aBars = false; s.split = null; s.hudL = false; s.panel = 'bowl'; },
        at(s, t) { s.eigGrow = seg(t, 0.1, 0.7, 'out'); } },

      { title: 'One step', duration: 3,
        narration: 'Start at the orange w₀ = (2.5, 1.5). The gradient ∇L = Hw₀ is drawn yellow, perpendicular to the contour (p5-01). One step of <b>gradient descent</b> moves against it: w₁ = w₀ − η∇L, with <b>learning rate</b> η = 0.15. The purple step is the yellow arrow reversed.',
        setup(s) { s.eigGrow = 1; s.showStart = true; s.k = 0; s.gradShow = 0; s.stepShow = 0; s.panel = 'step'; },
        at(s, t) { s.gradShow = seg(t, 0, 0.25, 'out'); s.stepShow = seg(t, 0.3, 0.75, 'inOut'); s.k = t > 0.8 ? 1 : 0; } },

      { title: 'Many steps', duration: 4,
        narration: 'Repeat. The orange path bounces across the narrow valley while creeping along it toward the gold minimum. Twelve steps and it is still not there. This bounce is the <b>zig-zag</b>.',
        setup(s) { s.gradShow = 0; s.stepShow = 0; s.k = 1; s.hudL = true; s.panel = 'many'; },
        at(s, t) { s.k = 1 + 11 * seg(t, 0.03, 0.97, 'linear'); } },

      { title: 'Why it bounces', duration: 4,
        narration: 'Each step is perpendicular to its contour, not aimed at the center (p5-01’s ✕). On a narrow bowl that direction is mostly across the valley: the step overshoots to the other wall, then overshoots back. The along-valley part of each step, drawn dim, is tiny.',
        setup(s) { s.k = 0; s.decomp = true; s.hudL = false; s.panel = 'why'; s.cam = cam2(); },
        at(s, t) { LA.cam.tween(s, { center: cam2().center, halfH: 3.6 }, CAMZ, seg(t, 0, 0.18, 'inOut')); s.k = 4 * seg(t, 0.2, 0.92, 'linear'); } },

      { title: 'The eigenbasis view', duration: 5,
        narration: 'Rotate the picture so the gold axes are the screen axes (a change of basis, p4-01). Now each coordinate shrinks on its own: a₁ ← (1 − 0.15·10)a₁ = −0.5a₁ flips and halves; a₂ ← (1 − 0.15·1)a₂ = 0.85a₂ barely moves. The bounce and the crawl are two independent 1D problems.',
        setup(s) { s.cam = cam2(); s.decomp = false; s.k = 0; s.eigView = 0; s.aBars = true; s.panel = 'eig'; },
        at(s, t) { s.eigView = seg(t, 0, 0.35, 'inOut'); s.k = 12 * seg(t, 0.4, 0.97, 'linear'); } },

      { title: 'The learning rate and the red line', duration: 6,
        narration: 'Slide η. Tiny η: slow crawl. Larger: faster, until the across-valley factor 1 − 10η passes −1 and the bounce grows instead of shrinking. That is the <b>stability limit</b>, η = 2/λmax = 0.2, drawn as the red line. Past it the path flies out of the bowl.',
        setup(s) { s.eigView = 0; s.aBars = false; s.etaAxis = true; s.factorBars = true; s.eta = 0.02; s.path = gd(Hof(s.lam, s.phi), s.w0, 0.02, 0, 20); s.k = 20; s.panel = 'eta'; },
        at(s, t) { s.eta = Math.round((0.02 + 0.24 * seg(t, 0.03, 0.97, 'linear')) * 1000) / 1000; s.path = gd(Hof(s.lam, s.phi), s.w0, s.eta, 0, 20); s.k = s.path.length - 1; } },

      { title: '✕ Smaller is not always better', duration: 4.5, misconception: true,
        narration: '“A small learning rate is always the safe, good choice.” Safe, yes; good, no. With η = 0.01, after 40 steps the along-valley coordinate is still 0.99⁴⁰ = 67% of where it started. Too small wastes every step; too large bounces. The bowl’s eigenvalues set the window.',
        setup(s) { s.etaAxis = false; s.factorBars = false; s.eta = ETA; s.split = 'mis'; s.path = gd(Hof(s.lam, s.phi), s.w0, 0.01, 0, 40); s.pathM = gd(Hof(s.lam, s.phi), s.w0, ETA, 0, 40); s.k = 0; s.panel = 'mis'; },
        at(s, t) { s.k = 40 * seg(t, 0.03, 0.8, 'linear'); } },

      { title: 'The condition number', duration: 4.5,
        narration: 'The best η balances both axes: η = 2/(λ₁ + λ₂) ≈ 0.18, where both factors have size (κ − 1)/(κ + 1) = 0.82. The steps needed grow with κ. Round the bowl — slide λ₂ up to 10 — and a single step lands at the center.',
        setup(s) { s.split = null; s.pathM = []; s.lam = LAM.slice(); s.eta = 2 / 11; s.path = gd(Hof(s.lam, s.phi), s.w0, s.eta, 0, 12); s.k = 12; s.panel = 'kappa'; },
        at(s, t) { const l2 = 1 + 9 * seg(t, 0.1, 0.9, 'inOut'); s.lam = [10, Math.round(l2 * 1000) / 1000]; s.eta = 2 / (10 + s.lam[1]); s.path = gd(Hof(s.lam, s.phi), s.w0, s.eta, 0, 12); s.k = 12; } },

      { title: 'Where elongation comes from', duration: 5,
        narration: 'For least squares the Hessian is XᵀX/n (p3-06): the spread of the features. Here feature 2 is spread 3× wider than feature 1, so H = diag(1, 9) and κ = 9. <b>Standardize</b> both features to unit spread and H becomes I: the bowl is round and one step finishes.',
        setup(s) { s.split = 'std'; s.lam = [9, 1]; s.phi = PI / 2; s.stdU = 0; s.eta = 0.2; s.path = gd(Hof(s.lam, s.phi), s.w0, 0.2, 0, 12); s.k = 12; s.panel = 'std'; },
        at(s, t) { const u = seg(t, 0.25, 0.85, 'inOut'); s.stdU = u; const l1 = 9 - 8 * u; s.lam = [Math.round(l1 * 1000) / 1000, 1]; s.eta = 2 / (s.lam[0] + 1); s.path = gd(Hof(s.lam, s.phi), s.w0, s.eta, 0, 12); s.k = 12; } },

      { title: 'In 3D', duration: 4,
        narration: 'The same path lifted onto the loss surface: a ball bouncing from wall to wall down a canyon. Orbit to see that every bounce lands higher on the walls than a straight descent would. Nothing new — only the view.',
        setup(s) { s.split = null; s.stdU = 0; s.lam = LAM.slice(); s.phi = PHI; s.eta = ETA; s.path = gd(Hof(LAM, PHI), s.w0, ETA, 0, 12); s.k = 12; s.view = '3d'; s.cam = cam3({ yaw: -1.6 }); s.panel = 'many'; },
        at(s, t) { LA.cam.tween(s, { yaw: -1.6 }, { yaw: -0.75 }, seg(t, 0, 1, 'linear')); s.k = 12 * seg(t, 0, 0.6, 'linear'); } },

      { title: 'Momentum', duration: 4.5,
        narration: 'Carry part of the previous step into the next: v ← βv − η∇L, w ← w + v, β = 0.5. This is <b>momentum</b>. Along the valley the pushes agree: the crawl speeds up from ×0.85 to about ×0.71 per step. Across it, the bounce shrinks a bit more slowly (×0.71 instead of ×0.5). The crawl was the bottleneck, so purple arrives first here.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.path = gd(Hof(LAM, PHI), s.w0, ETA, 0, 25); s.pathM = gd(Hof(LAM, PHI), s.w0, ETA, BETA_M, 25); s.k = 0; s.panel = 'mom'; },
        at(s, t) { s.k = 25 * seg(t, 0.03, 0.9, 'linear'); } },

      { title: 'Why ML cares: every training loop', duration: 4.5, mlTie: true,
        narration: 'Real losses are not quadratic, but near a minimum L ≈ ½(w − w*)ᵀH(w − w*) (p5-02), so this picture is the local truth. Standardizing inputs, normalization layers (p6-07), momentum and learning-rate warnings are all about this bowl’s eigenvalues. The picture stops at 3. The formula does not.',
        setup(s) { s.pathM = []; s.ml = 0.001; s.panel = 'ml'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0, 0], halfH: 3.9 }); },
        at(s, t) { s.ml = 0.001 + 0.999 * seg(t, 0, 1, 'linear'); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Gradient descent shrinks each eigen-coordinate by 1 − ηλᵢ. The largest curvature caps the step at 2/λmax; the smallest sets how slowly you crawl; their ratio κ is the whole story. Round the bowl and the story is short.',
        cards: ['<b>Each eigen-coordinate shrinks by 1 − ηλᵢ.</b> Across the valley it flips and halves; along it, it barely moves.', '<b>η < 2/λmax or it blows up.</b> The steepest curvature sets the red line; too small an η wastes steps.', '<b>κ decides the number of steps.</b> Standardizing features (or normalization) rounds the bowl.'],
        setup(s) { s.ml = 0; s.recap = true; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Slide η and β, or drag the orange start point and the knob on the η axis. The red line is 2/λmax; κ and the path update live.</p>',
      setup(s) { s.view = '2d'; s.cam = cam2(); s.lam = LAM.slice(); s.phi = PHI; s.w0 = W0.slice(); s.eta = ETA; s.beta = 0; s.steps = 20; s.etaAxis = true; s.factorBars = false; s.aBars = true; s.showStart = true; s.showEig = true; s.eigGrow = 1; s.panel = 'play'; },
      controls: [
        { type: 'slider', bind: 'eta', min: 0, max: 0.3, step: 0.005, label: 'Learning rate η', color: 'b', fmt: x => fmt(x, 3) },
        { type: 'slider', bind: 'beta', min: 0, max: 0.95, step: 0.05, label: 'Momentum β', color: 'w' },
        { type: 'slider', bind: 'lam[1]', min: 0.5, max: 10, step: 0.1, label: 'Small eigenvalue λ₂', color: 'gold' },
        { type: 'slider', bind: 'steps', min: 0, max: 60, step: 1, label: 'Steps', fmt: x => fmt(x, 0) },
        { type: 'preset', label: 'Bowl', items: [
          { label: 'Rotated κ = 10', apply(s) { s.lam = [10, 1]; s.phi = PHI; s.eta = ETA; s.w0 = W0.slice(); } },
          { label: 'Regression (raw)', apply(s) { s.lam = [9, 1]; s.phi = PI / 2; s.eta = 0.2; s.w0 = W0.slice(); } },
          { label: 'Regression (standardized)', apply(s) { s.lam = [1, 1]; s.phi = PI / 2; s.eta = 1; s.w0 = W0.slice(); } },
          { label: 'Round', apply(s) { s.lam = [10, 10]; s.phi = PHI; s.eta = 0.1; s.w0 = W0.slice(); } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2D map', apply(s) { s.view = '2d'; s.cam = cam2(); s.eigView = 0; } },
          { label: 'Eigenbasis view', apply(s) { s.view = '2d'; s.cam = cam2(); s.eigView = 1; } },
          { label: '3D surface', apply(s) { s.view = '3d'; s.cam = cam3(); s.eigView = 0; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the knob on the η axis so gradient descent diverges (after 20 steps it is more than 10× farther out than it started).',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.lam = LAM.slice(); s.phi = PHI; s.w0 = W0.slice(); s.eta = 0.1; s.beta = 0; s.steps = 20; s.etaAxis = true; s.factorBars = true; s.aBars = false; s.showStart = true; s.eigGrow = 1; s.panel = 'eta'; },
        judge(s) { const p = gd(Hof(s.lam, s.phi), s.w0, s.eta, 0, 20), r = N.norm(p[p.length - 1]) / N.norm(s.w0); return { ok: r > 10, progress: Math.min(1, Math.max(0, Math.log10(Math.max(r, 1e-9)) + 1) / 2) }; },
        hint: 'Cross the red line — and go a little past it: just over the line the bounce grows too slowly to reach 10× in 20 steps.',
        solution(s) { s.eta = 0.28; } },
      { kind: 'manipulate',
        prompt: 'Drag the η knob so the path gets within 0.2 of the minimum in at most 15 steps.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.lam = LAM.slice(); s.phi = PHI; s.w0 = W0.slice(); s.eta = 0.05; s.beta = 0; s.steps = 15; s.etaAxis = true; s.factorBars = true; s.aBars = false; s.showStart = true; s.eigGrow = 1; s.panel = 'eta'; },
        judge(s) { const p = gd(Hof(s.lam, s.phi), s.w0, s.eta, 0, 15), d = N.norm(p[p.length - 1]); return { ok: d < 0.2, progress: Math.min(1, 0.2 / Math.max(d, 1e-9)) }; },
        hint: 'Too slow along the valley: raise η. Bounce growing: lower it. The sweet spot is just under the red line.',
        solution(s) { s.eta = 0.18; } },
      { kind: 'manipulate',
        prompt: 'Drag the orange start point so gradient descent goes straight to the minimum with no zig-zag.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.lam = LAM.slice(); s.phi = PHI; s.w0 = W0.slice(); s.eta = ETA; s.beta = 0; s.steps = 12; s.etaAxis = false; s.factorBars = false; s.aBars = false; s.showStart = true; s.showEig = false; s.eigGrow = 0; s.panel = 'play'; },   // no gold axes: they would mark the answer
        judge(s) {
          const Q = qs(s.phi), n = N.norm(s.w0); if (n < 0.5) return { ok: false, progress: 0, hint: 'Move the start farther than 0.5 from the minimum.' };
          const ang = Math.min(...[Q[0], Q[1]].map(q => Math.acos(Math.min(1, Math.abs(N.dot(s.w0, q)) / n)) * 180 / PI));
          return { ok: ang < LA.TOL.angleDeg, progress: Math.max(0, 1 - ang / 45) };
        },
        hint: 'Start on a symmetry axis of the ellipses — one of the gold eigen-directions from step 1.',
        solution(s) { s.w0 = [2, 2]; } },
    ],
  });

  /* ---------------- shared pieces ---------------- */
  /** Draw a path revealed up to k (float): segments, dots, the current segment purple; returns the last drawn point. */
  function drawPath(g, pts, k, o) {
    o = o || {}; const T = o.T || (p => p), color = o.color || 'b', n = pts.length - 1;
    const kk = LA.clamp(k, 0, n), full = Math.floor(kk + 1e-9), frac = kk - full;
    const out = 8, vis = [];
    let flew = false;
    for (let i = 0; i <= full; i++) { if (N.norm(pts[i]) > out) { flew = true; break; } vis.push(T(pts[i])); }
    let tip = vis[vis.length - 1];
    if (!flew && frac > 1e-6 && full < n && N.norm(pts[full + 1]) <= out) tip = T(lerpV(pts[full], pts[full + 1], frac));
    const line = vis.slice(); if (tip !== vis[vis.length - 1]) line.push(tip);
    if (line.length > 1) g.polyline(line, { color, width: o.width || 2.4, alpha: o.alpha === undefined ? 1 : o.alpha });
    if (!o.noDots) vis.forEach((p, i) => g.point(p, { r: i === 0 ? 6 : 3.6, color, alpha: o.alpha === undefined ? 1 : o.alpha }));
    if (o.current && line.length > 1 && frac > 1e-6) g.vector(T(pts[full]), tip, { color: 'w', width: 3, head: 9 });
    if (flew && o.flyLabel !== false && vis.length) {
      const b = g.P.bounds ? g.P.bounds() : null, last = vis[vis.length - 1];
      const at = b ? [LA.clamp(last[0], b.x0 + 0.2 * (b.x1 - b.x0), b.x1 - 0.35 * (b.x1 - b.x0)), LA.clamp(last[1], b.y0 + 0.1 * (b.y1 - b.y0), b.y1 - 0.25 * (b.y1 - b.y0))] : last;
      g.label('flies out of the bowl', at, { anchor: 'left', size: 13, color: 'bad', font: 'ui' });
    }
    return { tip, flew, full };
  }
  function drawEig(g, s, T, grow) {
    const Q = qs(s.phi), lens = [Math.sqrt(8 / s.lam[0]), Math.sqrt(8 / s.lam[1])];
    Q.forEach((q, i) => {
      g.lineThrough(T([0, 0]), T(q), { len: 6, color: 'gold', width: 1, alpha: 0.4, dashed: true });
      if (grow > 0.01) g.vector(T([0, 0]), T(N.scale(q, Math.min(3, lens[i]) * grow)), { color: 'gold', width: 3, label: 'q' + (i ? '₂' : '₁') + ', λ = ' + fmt(s.lam[i]), labelAt: 1, labelSize: 12.5 });
    });
  }
  /** Screen → world helper (2D only). */
  const SW = g => (x, y) => g.P.toWorld(x, y);

  /** The η axis HUD (top-right) with the red 2/λmax line and a draggable knob. */
  function etaAxis(g, s, ctx, lmax) {
    const W = SW(g), x0 = ctx.size.w - 260, x1 = ctx.size.w - 30, y = 40, emax = 0.3;
    const X = e => x0 + (e / emax) * (x1 - x0);
    g.polygon([W(x0 - 14, y - 26), W(x1 + 18, y - 26), W(x1 + 18, y + 30), W(x0 - 14, y + 30)], { fill: 'panel', alpha: 0.85, stroke: 'line', width: 1 });
    g.line(W(x0, y), W(x1, y), { color: 'hud', width: 1.4 });
    [0, 0.1, 0.2, 0.3].forEach(e => { g.line(W(X(e), y - 4), W(X(e), y + 4), { color: 'hud', width: 1.2 }); g.label(fmt(e), [X(e), y + 16], { screen: true, size: 10.5, color: 'tick', font: 'ui', bg: false }); });
    const lim = 2 / lmax;
    if (s.showLine && lim <= emax) { g.line(W(X(lim), y - 16), W(X(lim), y + 10), { color: 'bad', width: 2.4 }); g.label('2/λmax = ' + fmt(lim), [X(lim), y - 19], { screen: true, size: 11, color: 'bad', font: 'ui', bg: false }); }
    const e = LA.clamp(s.eta, 0, emax), bad = s.eta > lim;
    g.label('η', [x1 + 10, y], { screen: true, size: 14, color: 'hud', font: 'math', bg: false });
    const kw = W(X(e), y);
    g.point(kw, { r: 7, color: bad ? 'bad' : 'b' });
    g.label('η = ' + fmt(s.eta, 3) + (bad ? ' (diverges)' : ''), [x0 - 4, y - 16], { screen: true, anchor: 'left', size: 11.5, color: bad ? 'bad' : 'b', font: 'ui', bg: false });
    const P = g.P;
    g.handle('eta', kw, (q, st) => { const sx = P.toScreen(q)[0]; st.eta = Math.round(LA.clamp((sx - x0) / (x1 - x0) * emax, 0, emax) * 1000) / 1000; }, { color: 'b' });
  }
  /** |1 − ηλᵢ| bars with the dashed line at 1. */
  function factorBars(g, s, ctx) {
    const W = SW(g), x = ctx.size.w - 150, y = ctx.size.h - 140, w = 96, h = 90;
    const f = [Math.abs(1 - s.eta * s.lam[0]), Math.abs(1 - s.eta * s.lam[1])];
    g.polygon([W(x - 14, y - 28), W(x + w + 40, y - 28), W(x + w + 40, y + h + 26), W(x - 14, y + h + 26)], { fill: 'panel', alpha: 0.85, stroke: 'line', width: 1 });
    g.bars(f.map(v => Math.min(v, 2)), { screen: true, x, y, w, h, max: 2, colors: f.map(v => (v > 1 ? 'bad' : 'gold')), labels: ['λ₁ = ' + fmt(s.lam[0]), 'λ₂ = ' + fmt(s.lam[1])], title: '|1 − ηλᵢ| per step' });
    f.forEach((v, i) => g.label(fmt(v), [x + (i + 0.5) * (w - 4) / 2 + i * 4, y + h - Math.min(v, 2) / 2 * h - 9], { screen: true, size: 11, color: 'hud', font: 'ui', bg: false }));
    g.line(W(x - 6, y + h / 2), W(x + w + 6, y + h / 2), { color: 'hud', dashed: [4, 4], width: 1.2 });
    g.label('1', [x + w + 12, y + h / 2], { screen: true, size: 11, color: 'hud', font: 'ui', bg: false });
  }
  /** |a₁| and |a₂| per step (two little bar charts, bottom-left). */
  function aBars(g, s, ctx, path, k) {
    const Q = qs(s.phi), n = Math.min(path.length - 1, Math.floor(k + 1e-9));
    const A1 = [], A2 = [], C1 = [];
    for (let i = 0; i <= Math.max(n, 0); i++) { const a1 = N.dot(path[i], Q[0]), a2 = N.dot(path[i], Q[1]); A1.push(Math.min(Math.abs(a1), 3)); C1.push(a1 >= 0 ? 'b' : 'neg'); A2.push(Math.min(Math.abs(a2), 3)); }
    const W = SW(g), x = 22, w = 170, h = 52, y1 = ctx.size.h - 2 * h - 70, y2 = ctx.size.h - h - 30;
    const m1 = Math.max(Math.abs(N.dot(path[0], Q[0])), 1e-6), m2 = Math.max(Math.abs(N.dot(path[0], Q[1])), 1e-6);
    g.polygon([W(x - 10, y1 - 24), W(x + w + 10, y1 - 24), W(x + w + 10, y2 + h + 12), W(x - 10, y2 + h + 12)], { fill: 'panel', alpha: 0.85, stroke: 'line', width: 1 });
    const nb = Math.max(13, Math.round(s.steps || 12) + 1), pad = arr => arr.concat(new Array(Math.max(0, nb - arr.length)).fill(0));
    g.bars(pad(A1), { screen: true, x, y: y1, w, h, max: Math.min(3, m1), colors: pad(C1).map(c => c || 'b'), gap: 2, title: '|a₁| across (orange +, violet −)' });
    g.bars(pad(A2), { screen: true, x, y: y2, w, h, max: Math.min(3, m2), color: 'gold', gap: 2, title: '|a₂| along the valley' });
  }

  /* ---------------- 2D map ---------------- */
  function render2d(g, s, ctx, H, path, k, live) {
    const Q = qs(s.phi), ang = -s.phi * s.eigView, T = p => (s.eigView ? rot(ang, p) : p), Ti = p => (s.eigView ? rot(-ang, p) : p);
    g.ghostGrid();
    if (s.eigView < 0.5) g.axes({ names: ['w₁', 'w₂'] });
    else { g.axes({ ticks: false, names: ['a₁ (across)', 'a₂ (along)'] }); g.lineThrough([0, 0], rot(ang, [1, 0]), { len: 6, color: 'muted', width: 1, alpha: 0.5 }); g.lineThrough([0, 0], rot(ang, [0, 1]), { len: 6, color: 'muted', width: 1, alpha: 0.5 }); g.label('w₁', rot(ang, [3.2, 0]), { size: 12, color: 'muted' }); g.label('w₂', rot(ang, [0, 3.2]), { size: 12, color: 'muted' }); }
    g.contours(q => Lq(H, Ti(q)), LV, CB, { res: 60, color: 'tick', width: 1.15, alpha: 0.9 });
    if (s.showEig) drawEig(g, s, T, s.eigGrow);
    g.point([0, 0], { r: 6, color: 'gold', label: 'w*', dx: -14, dy: 13 });
    const w0 = s.w0;
    // the momentum path (purple) is drawn under the plain path
    if (s.pathM && s.pathM.length > 1 && !live) {
      const rm = drawPath(g, s.pathM, k, { color: 'w', T });
      const rp = s.path[Math.min(s.path.length - 1, Math.floor(k))], dm = N.norm(s.pathM[Math.min(s.pathM.length - 1, Math.floor(k))]), dp = N.norm(rp);
      g.hud('plain GD (orange): ‖w‖ = ' + fmt(dp, 3), 14, 22, { color: 'b' });
      g.hud('momentum β = 0.5 (purple): ‖w‖ = ' + fmt(dm, 3), 14, 42, { color: 'w' });
      g.hud('after ' + Math.floor(k + 1e-9) + ' steps', 14, 62, { color: 'muted' });
    }
    let r = { tip: T(w0) };
    if (s.showStart) r = drawPath(g, path, k, { color: path.some(p => N.norm(p) > 6) ? 'bad' : 'b', T, current: !live && s.panel === 'many', alpha: s.decomp ? 0.3 : 1 });
    if (s.decomp) decomposition(g, s, H, path, k, T);
    if (s.gradShow > 0.01) { const gr = grad(H, w0); g.vector(T(w0), T(N.add(w0, N.scale(gr, ETA * s.gradShow))), { color: 'v', width: 3, label: '∇L (drawn ×0.15)', labelAt: 0.6, labelSize: 12 }); }
    if (s.stepShow > 0.01) { const gr = grad(H, w0); g.vector(T(w0), T(N.sub(w0, N.scale(gr, ETA * s.stepShow))), { color: 'w', width: 3.2, label: '−η∇L', labelAt: 0.5, labelSize: 12.5, dx: 4, dy: 18 }); }
    if (s.showStart) { g.label('w₀', T(w0), { dx: 14, dy: -12, size: 13, color: 'b' }); }
    if (s.hudL) { const kk = Math.min(path.length - 1, Math.floor(k + 1e-9)); g.hud('L(w' + subk(kk) + ') = ' + fmt(Lq(H, path[kk]), 3), 14, 22, { color: 'b' }); }
    if (s.aBars) aBars(g, s, ctx, path, live ? path.length - 1 : k);
    if (s.etaAxis) etaAxis(g, s, ctx, Math.max(s.lam[0], s.lam[1]));
    if (s.factorBars) factorBars(g, s, ctx);
    if (s.panel === 'kappa') g.hud('κ = ' + fmt(Math.max(s.lam[0], s.lam[1]) / Math.min(s.lam[0], s.lam[1])) + '   ·   η = 2/(λ₁ + λ₂) = ' + fmt(s.eta, 3), 14, 22, { color: 'gold' });
    if (s.showStart && s.panel === 'step' && k >= 1) g.label('w₁', T(path[1]), { dx: -16, dy: -10, size: 13, color: 'b' });
    if (s.showStart) g.handle('w0', T(w0), (q, st) => { const p = st.eigView ? rot(st.phi * st.eigView, q) : q; st.w0 = [LA.clamp(p[0], -3.2, 3.2), LA.clamp(p[1], -3.2, 3.2)]; }, { color: 'b' });
  }
  const subk = k => String(k).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c] || c).join('');

  /** Step 4: each of the first steps split into its q₁ (across, bright) and q₂ (along, dim) parts. */
  function decomposition(g, s, H, path, k, T) {
    const Q = qs(s.phi), n = Math.min(4, Math.floor(k + 1e-9));
    for (let i = 0; i < n; i++) {
      const a = path[i], b = path[i + 1], d = N.sub(b, a), c1 = N.dot(d, Q[0]), c2 = N.dot(d, Q[1]);
      const m = N.add(a, N.scale(Q[0], c1));
      g.vector(T(a), T(m), { color: 'b', width: 3.6, head: 11, label: i === 0 ? 'across (big, flips)' : undefined, labelAt: 0.5, labelSize: 12 });
      g.vector(T(m), T(b), { color: 'gold', width: 2.2, head: 8, alpha: 0.55, label: i === 0 ? 'along (small)' : undefined, labelAt: 0.5, labelSize: 12 });
      g.point(T(b), { r: 3.6, color: 'b' });
    }
    g.point(T(path[0]), { r: 6, color: 'b' });
    if (n >= 1) {
      const p0 = N.scale(Q[1], N.dot(path[0], Q[1])), pn = N.scale(Q[1], N.dot(path[n], Q[1]));
      g.dim(T(p0), T(pn), { label: 'along-valley progress ' + fmt(N.dot(path[0], Q[1]) - N.dot(path[n], Q[1])), offset: 26, color: 'gold' });
    }
  }

  /* ---------------- split views ---------------- */
  function renderSplitMis(g, s) {
    const H = Hof(s.lam, s.phi), cam = { center: [1.0, 0.95], halfH: 1.75 };
    const pane = (y, path, eta, label, bad) => g.viewport({ x: 0, y, w: 1, h: 0.5 }, { view: '2d', cam }, (gg, r) => {
      gg.ghostGrid();
      gg.contours(q => Lq(H, q), LV, CB, { res: 56, color: 'tick', width: 1.1 });
      drawEig(gg, s, p => p, 0);
      gg.point([0, 0], { r: 6, color: 'gold', label: 'w*', dx: -14, dy: 13 });
      drawPath(gg, path, s.k, { color: 'b' });
      gg.label('η = ' + fmt(eta) + ', ' + Math.floor(s.k + 1e-9) + ' steps', [r.x + 14, r.y + 18], { screen: true, anchor: 'left', size: 13.5, color: 'b', font: 'ui', bold: true, bg: false });
      const Q = qs(s.phi), kk = Math.min(path.length - 1, Math.floor(s.k + 1e-9)), d = Math.abs(N.dot(path[kk], Q[1])), d0 = Math.abs(N.dot(path[0], Q[1]));
      const bx = r.x + 24, by = r.y + r.h - 112;
      gg.bars([d], { screen: true, x: bx, y: by, w: 26, h: 80, max: d0, colors: [bad ? 'bad' : 'ok'], values: true, title: 'along-valley coordinate left' });
      gg.label(fmt(100 * d / d0, 1) + '% of the start', [bx + 36, by + 60], { screen: true, anchor: 'left', size: 12, color: bad ? 'bad' : 'ok', font: 'ui', bg: false });
      if (bad) { const lp = [r.x + 34, r.y + 44]; gg.label('“smaller is better”', lp, { screen: true, anchor: 'left', size: 13, color: 'muted', font: 'ui' }); if (s.k > 30) gg.marker.cross(lp, { screen: true, size: 20, dx: -16 }); }
    });
    pane(0, s.path, 0.01, 'tiny', true);
    pane(0.5, s.pathM, ETA, 'right', false);
  }

  function renderSplitStd(g, s) {
    const u = s.stdU, H = Hof(s.lam, s.phi);
    g.viewport({ x: 0, y: 0, w: 1, h: 0.42 }, { view: '2d', cam: { center: [0, 0], halfH: 10 } }, (gg, r) => {
      gg.line([-30, 0], [30, 0], { color: 'gridAxis', width: 1 }); gg.line([0, -12], [0, 12], { color: 'gridAxis', width: 1 });
      const sy = 1 + (1 / 3 - 1) * u;
      gg.points(s.data.map(p => [p[0], p[1] * sy]), { r: 3.6, color: 'data' });
      gg.label('feature 1', [r.x + r.w / 2 + 120, r.y + r.h / 2 + 14], { screen: true, size: 11.5, color: 'muted', font: 'ui', bg: false });
      gg.label('feature 2', [r.x + r.w / 2 + 34, r.y + 14], { screen: true, size: 11.5, color: 'muted', font: 'ui', bg: false });
      gg.label(u > 0.98 ? 'standardized: spread 1 × 1' : 'raw: spread 1 × 3 (synthetic data)', [r.x + 14, r.y + 18], { screen: true, anchor: 'left', size: 13, color: 'data', font: 'ui', bold: true, bg: false });
    });
    g.viewport({ x: 0, y: 0.42, w: 1, h: 0.58 }, { view: '2d', cam: { center: [1.1, 0.4], halfH: 2.1 } }, (gg, r) => {
      gg.ghostGrid(); gg.axes({ names: ['w₁', 'w₂'] });
      gg.contours(q => Lq(H, q), LV, CB, { res: 56, color: 'tick', width: 1.1 });
      gg.point([0, 0], { r: 6, color: 'gold', label: 'w*', dx: -14, dy: 13 });
      drawPath(gg, s.path, s.k, { color: 'b' });
      gg.label('H = diag(' + fmt(s.lam[1]) + ', ' + fmt(s.lam[0]) + ') · κ = ' + fmt(s.lam[0] / s.lam[1]) + ' · η = ' + fmt(s.eta, 2), [r.x + 14, r.y + 18], { screen: true, anchor: 'left', size: 13, color: 'b', font: 'ui', bold: true, bg: false });
    });
  }

  /* ---------------- 3D ---------------- */
  function drawSurface(g, H, alpha) {
    g.paramSurface((uu, v) => {
      const d = [Math.cos(v), Math.sin(v)], kq = H[0][0] * d[0] * d[0] + 2 * H[0][1] * d[0] * d[1] + H[1][1] * d[1] * d[1];
      const rm = kq > 1e-6 ? Math.min(RMAX, Math.sqrt(2 * CAP / kq)) : RMAX, x = uu * rm * d[0], y = uu * rm * d[1];
      return [x, y, ZS * Lq(H, [x, y])];
    }, { u0: 0, u1: 1, v0: 0, v1: 2 * PI }, { res: 24, mode: 'both', color: 'grid', alpha: 0.24 * alpha, wireColor: 'gridAxis', wireAlpha: 0.4 * alpha });
  }
  function render3d(g, s, ctx, H, path, k) {
    g.cage({ range: 3, walls: false, alpha: 0.6 });
    g.contours(q => Lq(H, q), LV, CB, { res: 44, color: 'tick', width: 1.05, alpha: 0.85, z: 0 });
    drawSurface(g, H, 1);
    g.axes({ length: 3.4, names: ['w₁', 'w₂', 'L'] });
    g.point([0, 0, 0], { r: 6, color: 'gold', label: 'w*', dx: -14, dy: 13 });
    const n = Math.min(path.length - 1, Math.floor(k + 1e-9)), lift = p => [p[0], p[1], ZS * Lq(H, p) + 0.02];
    const vis = []; for (let i = 0; i <= n; i++) { if (N.norm(path[i]) > 4) break; vis.push(path[i]); }
    if (vis.length > 1) { g.polyline(vis.map(lift), { color: 'b', width: 2.6 }); g.polyline(vis.map(p => [p[0], p[1], 0]), { color: 'b', width: 1.2, alpha: 0.35 }); }
    vis.forEach((p, i) => { g.point(lift(p), { r: i === 0 ? 6 : 3.8, color: 'b' }); if (i === 0) dash3(g, [p[0], p[1], 0], lift(p), { color: 'b', width: 1, alpha: 0.6 }); });
    if (vis.length) g.label('w₀', lift(vis[0]), { dx: 14, dy: -12, size: 13, color: 'b' });
    g.handle('w0', [s.w0[0], s.w0[1], 0], (q, st) => { st.w0 = [LA.clamp(q[0], -3.2, 3.2), LA.clamp(q[1], -3.2, 3.2)]; }, { color: 'b', plane: 'xy' });
  }

  /* ---------------- mlTie ---------------- */
  function renderML(g, s) {
    const t = s.ml, H = Hof(LAM, PHI), cam = { center: [1.15, 0.85], halfH: 2.1 };
    const thumbs = [
      ['zig-zag (κ = 10)', H, gd(H, W0, ETA, 0, 12), 'b'],
      ['standardized (κ = 1)', Hof([1, 1], PHI), gd(Hof([1, 1], PHI), W0, 1, 0, 3), 'b'],
      ['momentum β = 0.5', H, gd(H, W0, ETA, BETA_M, 12), 'w'],
    ];
    thumbs.forEach((th, i) => {
      const a = seg(t, 0.05 + 0.12 * i, 0.3 + 0.12 * i);
      g.viewport({ x: 0.02 + i * 0.327, y: 0.03, w: 0.31, h: 0.34 }, { view: '2d', cam, bg: 'panel' }, (gg, r) => {
        gg.contours(q => Lq(th[1], q), LV, CB, { res: 44, color: 'tick', width: 1 });
        gg.point([0, 0], { r: 4.5, color: 'gold' });
        drawPath(gg, th[2], (th[2].length - 1) * a, { color: th[3], width: 2, noDots: false });
        gg.label(th[0], [r.x + 8, r.y + 14], { screen: true, anchor: 'left', size: 12, color: 'text', font: 'ui', bold: true, bg: false });
      });
    });
    const fade = seg(t, 0.35, 0.65);
    if (fade > 0.01) {
      const bx0 = -3.1, bx1 = 3.1, by0 = -0.15, by1 = 0.3, n = 12, cw = (bx1 - bx0) / n;
      g.heat2d([s.strip], { x0: bx0, x1: bx1, y0: by0, y1: by1 }, { range: [-1, 1], ramp: x => LA.color.diverge(2 * x - 1), smooth: false, alpha: 0.95 * fade });
      for (let i = 0; i <= n; i++) g.line([bx0 + i * cw, by0], [bx0 + i * cw, by1], { color: 'line', width: 1, alpha: fade });
      g.line([bx0, by0], [bx1, by0], { color: 'line', alpha: fade }); g.line([bx0, by1], [bx1, by1], { color: 'line', alpha: fade });
      for (let i = 0; i < n; i++) g.label(fmt(s.strip[i]), [bx0 + (i + 0.5) * cw, (by0 + by1) / 2], { size: 10, color: 'text', bg: false, font: 'ui', alpha: fade });
      g.label('∇L ∈ ℝ¹² (synthetic, seeded)', [bx0, by1], { anchor: 'left', dy: -12, size: 12, color: 'v', font: 'ui', alpha: fade });
      const hb = { x0: -3.1, x1: -0.7, y0: -3.1, y1: -0.7 }, M = s.heat6, m = Math.max(...M.map(r => Math.max(...r.map(Math.abs)))), hc = (hb.x1 - hb.x0) / 6;
      g.heat2d(M, hb, { range: [-m, m], ramp: x => LA.color.diverge(2 * x - 1), smooth: false, alpha: 0.95 * fade });
      for (let i = 0; i <= 6; i++) { g.line([hb.x0 + i * hc, hb.y0], [hb.x0 + i * hc, hb.y1], { color: 'line', width: 1, alpha: fade }); g.line([hb.x0, hb.y0 + i * hc], [hb.x1, hb.y0 + i * hc], { color: 'line', width: 1, alpha: fade }); }
      g.label('a 6×6 Hessian H (synthetic, seeded)', [(hb.x0 + hb.x1) / 2, hb.y1 + 0.25], { size: 11.5, color: 'hud', font: 'ui', alpha: fade });
      g.label('each eigen-direction of H', [0.0, -1.2], { anchor: 'left', size: 12.5, color: 'gold', font: 'ui', alpha: fade });
      g.label('shrinks by its own 1 − ηλᵢ', [0.0, -1.55], { anchor: 'left', size: 12.5, color: 'gold', font: 'ui', alpha: fade });
      g.label('capstone p6-08: this path', [0.0, -2.3], { anchor: 'left', size: 12, color: 'b', font: 'ui', alpha: fade });
      g.label('for a real classifier', [0.0, -2.62], { anchor: 'left', size: 12, color: 'b', font: 'ui', alpha: fade });
    }
    if (t > 0.6) g.label('The picture stops at 3. The formula does not.', [0, -3.5], { size: 13.5, color: 'hud', font: 'ui', alpha: seg(t, 0.6, 0.8) });
  }

  /* ---------------- recap ---------------- */
  function renderRecap(g, s) {
    const H = Hof(LAM, PHI);
    const card = (i, title, cam, fn) => g.viewport({ x: 0.04, y: 0.02 + i * 0.33, w: 0.92, h: 0.31 }, { view: '2d', cam, bg: 'panel' }, (gg, r) => { fn(gg, r); gg.label(title, [r.x + 12, r.y + 16], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true, bg: false }); });
    card(0, '1 · Each eigen-coordinate shrinks by 1 − ηλᵢ', { center: [1.6, 0.8], halfH: 1.6 }, gg => {
      gg.contours(q => Lq(H, q), LV, CB, { res: 50, color: 'tick', width: 1 });
      drawEig(gg, { lam: LAM, phi: PHI }, p => p, 0);
      gg.point([0, 0], { r: 5, color: 'gold' });
      drawPath(gg, gd(H, W0, ETA, 0, 12), 12, { color: 'b', width: 2 });
      gg.label('across: ×(−0.5)   along: ×0.85', [4.2, 0.5], { size: 12.5, color: 'gold', font: 'ui' });
    });
    card(1, '2 · η < 2/λmax or it blows up', { center: [0, 0], halfH: 1.2 }, gg => {
      const X = e => -2.8 + e / 0.3 * 5.6;
      gg.line([X(0), -0.2], [X(0.3), -0.2], { color: 'hud', width: 1.6 });
      [0, 0.1, 0.2, 0.3].forEach(e => { gg.line([X(e), -0.27], [X(e), -0.13], { color: 'hud' }); gg.label(fmt(e), [X(e), -0.45], { size: 11, color: 'tick', font: 'ui', bg: false }); });
      gg.polygon([[X(0.005), -0.12], [X(0.2), -0.12], [X(0.2), 0.12], [X(0.005), 0.12]], { fill: 'ok', alpha: 0.18 });
      gg.line([X(0.2), -0.55], [X(0.2), 0.45], { color: 'bad', width: 2.6 });
      gg.label('2/λmax = 0.2', [X(0.2), 0.62], { size: 12.5, color: 'bad', font: 'ui' });
      gg.label('converges (best near 0.18)', [X(0.1), 0.3], { size: 11.5, color: 'ok', font: 'ui' });
      gg.label('diverges', [X(0.26), 0.3], { size: 11.5, color: 'bad', font: 'ui' });
      gg.label('too small: crawls', [X(0.02), -0.75], { size: 11, color: 'muted', font: 'ui' });
    });
    card(2, '3 · κ decides the steps; standardize to round the bowl', { center: [0, 0], halfH: 1.25 }, gg => {
      gg.ellipse([[0.32, 0], [0, 0.95]], [-2.2, -0.1], { stroke: 'tick', width: 1.6 });
      gg.ellipse([[0.32 * 0.6, 0], [0, 0.95 * 0.6]], [-2.2, -0.1], { stroke: 'tick', width: 1.2 });
      gg.vector([-1.3, -0.1], [0.4, -0.1], { color: 'hud', width: 2, head: 10, label: 'standardize', labelAt: 0.5, labelSize: 12 });
      gg.circle([1.8, -0.1], 0.75, { stroke: 'tick', width: 1.6 }); gg.circle([1.8, -0.1], 0.45, { stroke: 'tick', width: 1.2 });
      gg.label('κ = 9: many steps', [-2.2, -1.05], { size: 12, color: 'b', font: 'ui' });
      gg.label('κ = 1: one step', [1.8, -1.05], { size: 12, color: 'ok', font: 'ui' });
      gg.vector([2.4, 0.4], [1.85, -0.07], { color: 'b', width: 2.2, head: 8 });
    });
  }
})();
