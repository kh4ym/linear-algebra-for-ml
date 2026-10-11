/* p5-02 — The Hessian and curvature. Slopes of slopes fill a symmetric 2×2 box; its eigenvectors are the bowl's
   sharpest and gentlest bends. f(x) = ½xᵀHx everywhere, so ∇f = Hx and the Hessian is H at every point. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, lerpM = LA.lerpM, seg = LA.seg, fmt = LA.fmt;
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const PI = Math.PI;
  const SC = 0.25;                                       // world units drawn per unit of slope
  const SCC = 0.25;                                      // scale of the column (∇f at î, ĵ) handles in check 3
  const CB = { x0: -3.6, x1: 3.6, y0: -3.6, y1: 3.6 };
  const H0 = [[2, 0], [0, 8]], HR = [[5, 3], [3, 5]], HS = [[2, 0], [0, -2]], HI = [[1, 0], [0, 1]];
  const LV0 = [0.5, 1, 2, 3, 4, 6, 8, 12], LVR = [1, 2, 4, 8, 12], LVS = [-4, -2, -1, 1, 2, 4];
  const LVC = [-4, -2, -1, -0.5, 0.5, 1, 2, 4, 8];
  const Q1 = [Math.SQRT1_2, Math.SQRT1_2], Q2 = [Math.SQRT1_2, -Math.SQRT1_2];
  const ZS = 0.25, CAP = 8, RMAX = 2.6;
  /** Dashed 3D segment drawn as short solid pieces (canvas line dashes on some 3D segments blanked headless frames). */
  function dash3(g, a, b, o) { const n = 18; for (let i = 0; i < n; i += 2) g.line(lerpV(a, b, i / n), lerpV(a, b, (i + 1) / n), o); }                  // 3D: height scale, cup rim level, max radius

  const fQ = (H, p) => 0.5 * (H[0][0] * p[0] * p[0] + 2 * H[0][1] * p[0] * p[1] + H[1][1] * p[1] * p[1]);
  const gQ = (H, p) => [H[0][0] * p[0] + H[0][1] * p[1], H[0][1] * p[0] + H[1][1] * p[1]];
  const symm = H => { const o = (H[0][1] + H[1][0]) / 2; return [[H[0][0], o], [o, H[1][1]]]; };
  const curvU = (H, u) => H[0][0] * u[0] * u[0] + 2 * H[0][1] * u[0] * u[1] + H[1][1] * u[1] * u[1];
  const unit = deg => [Math.cos(deg * PI / 180), Math.sin(deg * PI / 180)];
  const cam2 = () => Object.assign(LA.cam.defaults2d(), { halfH: 4 });
  const cam3 = o => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.3], dist: 13.4, yaw: -1.9, pitch: 0.6, fov: 36 }, o || {});
  const clampP = q => [LA.clamp(q[0], -3, 3), LA.clamp(q[1], -3, 3)];
  const goldNeg = () => LA.color.mix(LA.color.gold, LA.color.neg, 0.55);
  /** eigSym: values descending, vectors = array of unit, sign-fixed eigenvectors. */
  function eigVecs(H) { const e = N.eigSym(symm(H)); return { values: e.values, vecs: e.vectors }; }
  // Engine gap: tex.reconcile mis-assigns tex.num values when a panel also holds a tex.mat/vec. Numbers as plain
  // colored spans are part of the skeleton, so the panel just rebuilds.
  const numH = (tex, x, o) => { o = o || {}; return '<span class="tx-n ' + tex.colorClass(o.color) + '">' + (typeof x === 'number' ? fmt(x, o.digits) : x) + '</span>'; };
  function makeHeat6() {
    const rand = LA.rng(502), B = [];
    for (let i = 0; i < 6; i++) { const r = []; for (let j = 0; j < 6; j++) r.push(LA.rng.gauss(rand)); B.push(r); }
    const A = N.scaleM(N.matMul(N.transpose(B), B), 1 / 6);
    return A.map(r => r.map(x => Math.round(x * 100) / 100));
  }

  LA.registerLesson({
    id: 'p5-02-hessian-curvature',
    part: 5,
    title: 'The Hessian and curvature',
    subtitle: 'Slopes of slopes fill a symmetric 2×2 box; its eigenvectors are the sharpest and gentlest bends.',
    minutes: 10,
    prereqs: ['p5-01-gradient', 'p4-06-symmetric-quadratic-forms', 'p4-02-eigenvectors'],
    tags: ['core'],
    badge: { drawnIn: '2D contours + 3D surface over two inputs', realSize: 'the Hessian of a 10⁶-weight network is 10⁶×10⁶ — never stored, only products Hv', transfers: 'H is symmetric, uᵀHu is the curvature along u, eigenvectors are principal curvature directions, λmax sets the step limit — exact in any n' },
    cantShow: ['n inputs have n principal curvatures; we draw two.', 'For non-quadratic f the Hessian changes from point to point; we draw quadratics where it is the same everywhere.', 'Indefinite Hessians in n-D can bend down along many directions at once; the saddle shows one.'],
    terms: [
      { term: 'second derivative', def: 'The slope of the slope: how fast the slope changes as you move.' },
      { term: 'mixed partial', def: 'How the x-slope changes when you move along y, written ∂²f/∂x∂y; for smooth f it equals ∂²f/∂y∂x.' },
      { term: 'Hessian', def: 'The symmetric matrix H of all second derivatives of f.' },
      { term: 'curvature along u', def: 'The second slope along a unit direction u, equal to uᵀHu.' },
      { term: 'principal curvature directions', def: 'The eigenvectors of H, where the curvature is largest and smallest.' },
      { term: 'positive definite', def: 'All eigenvalues of H are positive: the surface is a bowl.' },
      { term: 'saddle', def: 'A surface whose Hessian has eigenvalues of both signs: up one way, down another.' },
    ],
    atlas: ['optimization', 'backprop'],
    thumbnail: 6,

    state() {
      return {
        view: '2d', cam: cam2(),
        H: LA.clone(H0), p: [1, 0.5], probe: 0, levels: LV0.slice(), guessH: [[0, 0], [0, 0]], heat6: makeHeat6(),
        arrX: 0, arrY: 0, alphaY: 1, trail: false, inset: null, box: -1, eigGrow: 0, eigLevel: null, showEig: false,
        showProbe: false, gauge: false, showGrad: false, showP: true, mis: 0, misCross: false, surfA: 1, ml: 0,
        colHandles: false, guessMode: false, recap: false, panel: 'd2x', axisNames: ['x', 'y'],
      };
    },

    render(g, s, ctx) { render0(g, s, ctx); },

    math(s, tex) {
      tex = Object.assign({}, tex, { num: (x, o) => numH(tex, x, o) });
      const H = s.H, gr = gQ(H, s.p), T = str => tex.text(str, { html: true });
      const Hs = tex.sym('H'), U = tex.sym('u', { color: 'w' });
      const hmat = M => tex.mat(M, { colors: (i, j) => (i === j ? (i === 0 ? 'i' : 'j') : '') });
      const ev = eigVecs(H), l1 = ev.values[0], l2 = ev.values[1];
      switch (s.panel) {
        case 'd2x': return tex.block(tex.eq(T('∂f/∂x'), tex.op('='), T(fmt(H[0][0]) + 'x'), tex.op('='), tex.num(gr[0], { color: 'i' })), tex.eq(tex.op('→'), T('∂²f/∂x²'), tex.op('='), tex.num(H[0][0], { color: 'i' })));
        case 'd2y': return tex.block(tex.eq(T('∂f/∂y'), tex.op('='), T(fmt(H[1][1]) + 'y'), tex.op('='), tex.num(gr[1], { color: 'j' })), tex.eq(tex.op('→'), T('∂²f/∂y²'), tex.op('='), tex.num(H[1][1], { color: 'j' })));
        case 'mixed': return tex.block(tex.eq(T('∂f/∂x'), tex.op('='), tex.num(gr[0], { color: 'i' }), tex.text('(unchanged as y moves)')), tex.eq(T('∂²f/∂x∂y'), tex.op('='), tex.num(0)));
        case 'H': return tex.block(tex.eq(Hs, tex.op('='), tex.mat([[{ html: '∂²f/∂x²' }, { html: '∂²f/∂x∂y' }], [{ html: '∂²f/∂y∂x' }, { html: '∂²f/∂y²' }]]), tex.op('='), hmat(H)));
        case 'form': return tex.lines(tex.eq(tex.sym('f'), tex.op('('), tex.sym('x', { bold: true }), tex.op(')'), tex.op('='), T('½ x<sup>ᵀ</sup>Hx'), tex.op('='), T('½(2x² + 8y²)'), tex.op('='), T('x² + 4y²')), tex.eq(Hs, tex.op('='), hmat(H)));
        case 'tilt': return tex.lines(tex.eq(Hs, tex.op('='), hmat(H)), tex.eq(T('∂f/∂x'), tex.op('='), T(fmt(H[0][0]) + 'x + ' + fmt(H[0][1]) + 'y'), tex.op('='), tex.num(gr[0], { color: 'i' }), tex.op('·'), T('∂²f/∂x∂y'), tex.op('='), tex.num(H[0][1])));
        case 'eig': return tex.lines(tex.eq(Hs, tex.op('='), T('QΛQ<sup>ᵀ</sup>'), tex.op('='), hmat(H)), tex.eq(T('λ₁'), tex.op('='), tex.num(l1, { color: 'gold' }), tex.op(','), T('λ₂'), tex.op('='), tex.num(l2, { color: 'gold' })));
        case 'curv': { const u = unit(s.probe); return tex.block(tex.eq(tex.text('curvature along'), U, tex.op('='), T('<span class="c-w">u</span><sup>ᵀ</sup>H<span class="c-w">u</span>'), tex.op('='), tex.num(curvU(H, u), { color: 'w' })), tex.eq(tex.text('between'), tex.num(l2, { color: 'gold' }), tex.text('and'), tex.num(l1, { color: 'gold' }))); }
        case 'mis': return tex.lines(tex.eq(tex.sym('∇f', { color: 'v' }), tex.op('='), T('Hx'), tex.text('changes with x'), tex.op('·'), Hs, tex.text('does not')), tex.eq(T('‖∇f(A)‖'), tex.op('='), tex.num(4, { color: 'v' }), tex.text('bend 2'), tex.op('·'), T('‖∇f(B)‖'), tex.op('='), tex.num(2.4, { color: 'v' }), tex.text('bend 8')));
        case 'saddle': return tex.lines(tex.eq(Hs, tex.op('='), hmat(H)), tex.eq(T('λ₁'), tex.op('='), tex.num(l1, { color: 'gold' }), tex.op(','), T('λ₂'), tex.op('='), tex.num(l2, { color: l2 < 0 ? 'neg' : 'gold' }), l2 < -0.05 ? tex.text('< 0 ⇒ saddle', { color: 'neg' }) : tex.text('')));
        case 'ml': return tex.lines(tex.eq(T('L(w) ≈ L(w*) + ½(w − w*)<sup>ᵀ</sup>H(w − w*)')), tex.eq(tex.text('safe:'), T('η < 2/λ<sub>max</sub>'), tex.op('='), tex.num(0.25, { color: 'bad' })));
        case 'guess': { const G = symm(s.guessH); return tex.lines(tex.eq(tex.sym('f'), tex.op('='), T('3x² + 2xy + y²')), tex.eq(tex.text('your'), Hs, tex.op('='), tex.mat(G, { digits: 2 }), tex.text('(only the symmetric part shapes f)'))); }
        case 'none': return '';
        default: {
          const u = unit(s.probe);
          return tex.lines(tex.eq(Hs, tex.op('='), hmat(H), tex.op('·'), T('λ'), tex.op('='), tex.num(l1, { color: 'gold' }), tex.op(','), tex.num(l2, { color: l2 < 0 ? 'neg' : 'gold' })),
            tex.eq(tex.sym('∇f', { color: 'v' }), tex.op('='), T('Hp'), tex.op('='), tex.vec(gr, { color: 'v' }), tex.op('·'), T('<span class="c-w">u</span><sup>ᵀ</sup>H<span class="c-w">u</span>'), tex.op('='), tex.num(curvU(H, u), { color: 'w' })));
        }
      }
    },

    steps: [
      { title: 'The slope changes', duration: 3,
        narration: 'Last lesson the slope along the green ' + ihat + ' at (1, 0.5) was 2. Move right and it grows: at x = 2 it is 4, at x = 3 it would be 6. The slope itself has a slope — here 2 per unit of x. That number is the <b>second derivative</b>.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.H = LA.clone(H0); s.levels = LV0.slice(); s.arrX = 1; s.arrY = 0; s.alphaY = 1; s.trail = false; s.inset = 'x'; s.box = -1; s.showEig = false; s.eigGrow = 0; s.eigLevel = null; s.showProbe = false; s.gauge = false; s.showGrad = false; s.mis = 0; s.panel = 'd2x'; s.p = [-2, 0.5]; },
        at(s, t) { s.p = [LA.lerp(-2, 2, seg(t, 0.05, 0.95, 'inOut')), 0.5]; } },

      { title: 'A steeper bend', duration: 3,
        narration: 'Along the red ' + jhat + ' the slope is 8y, so it changes four times faster: ∂²f/∂y² = 8. In the inset the red y-parabola is narrower than the green x-parabola. Curvature is “how fast the slope changes”, one number per direction.',
        setup(s) { s.arrX = 0; s.arrY = 1; s.inset = 'xy'; s.panel = 'd2y'; s.p = [1, -1]; },
        at(s, t) { s.p = [1, LA.lerp(-1, 1, seg(t, 0.05, 0.95, 'inOut'))]; } },

      { title: 'Does the x-slope care about y?', duration: 3,
        narration: 'Move along the red ' + jhat + ' again and watch the green arrow: it does not change. For this bowl the x-slope ignores y, so the <b>mixed partial</b> ∂²f/∂x∂y is 0. Soon we meet a bowl where it is not.',
        setup(s) { s.arrX = 1; s.arrY = 1; s.alphaY = 0.3; s.trail = true; s.inset = null; s.panel = 'mixed'; s.p = [1, -1]; },
        at(s, t) { s.p = [1, LA.lerp(-1, 1, seg(t, 0.05, 0.95, 'inOut'))]; } },

      { title: 'The 2×2 box', duration: 3,
        narration: 'Four second derivatives fit a 2×2 box: the <b>Hessian</b> <b>H</b>. The two mixed partials are always equal, so H is symmetric — the nice case from p4-06. For our bowl H = [[2, 0], [0, 8]].',
        setup(s) { s.trail = false; s.alphaY = 1; s.arrX = 1; s.arrY = 1; s.p = [1, 0.5]; s.inset = 'box'; s.box = 0; s.panel = 'H'; },
        at(s, t) { s.box = seg(t, 0.05, 0.6, 'inOut') + (t > 0.6 ? seg(t, 0.6, 1, 'linear') : 0); } },

      { title: 'A bowl is ½xᵀHx', duration: 3.5,
        narration: 'Check: ½xᵀHx = ½(2x² + 8y²) = x² + 4y². Our bowl is exactly the quadratic form from p4-06 with A = H. Near any minimum every smooth loss looks like this, so the Hessian is the bowl’s shape. The gold ellipse is the level f = 2.',
        setup(s) { s.box = -1; s.inset = null; s.arrX = 0; s.arrY = 0; s.view = '3d'; s.cam = cam3({ yaw: -1.9 }); s.surfA = 1; s.eigLevel = 2; s.eigGrow = 1; s.showEig = true; s.panel = 'form'; s.showP = false; },
        at(s, t) { LA.cam.tween(s, { yaw: -1.9 }, { yaw: -1.35 }, seg(t, 0, 1, 'linear')); } },

      { title: 'A tilted bowl', duration: 4,
        narration: 'Now H = [[5, 3], [3, 5]]. The contours tilt and the mixed partial is 3: moving along the red ' + jhat + ' now changes the green ' + ihat + '-slope. The slope along ' + ihat + ' is 5x + 3y — it depends on both coordinates.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.showEig = false; s.eigGrow = 0; s.eigLevel = null; s.levels = LVR.slice(); s.showP = true; s.arrX = 1; s.arrY = 0; s.p = [1, -1]; s._H = LA.clone(s.H); s.panel = 'tilt'; },
        at(s, t) { s.H = lerpM(s._H, HR, seg(t, 0, 0.45, 'inOut')); s.p = [1, LA.lerp(-1, 1, seg(t, 0.5, 0.97, 'inOut'))]; } },

      { title: 'Principal directions', duration: 4,
        narration: 'Where does the tilted bowl bend most? Along the eigenvectors of H (p4-02): gold q₁ = (1, 1)/√2 with λ₁ = 8 is the steep bend, gold q₂ = (1, −1)/√2 with λ₂ = 2 the gentle one. These are the <b>principal curvature directions</b>. Both λ positive: the bowl is <b>positive definite</b>.',
        setup(s) { s.H = LA.clone(HR); s.arrX = 0; s.showP = false; s.showEig = true; s.eigGrow = 0; s.eigLevel = 4; s.inset = 'eig'; s.panel = 'eig'; },
        at(s, t) { s.eigGrow = seg(t, 0.05, 0.6, 'out'); } },

      { title: 'Curvature in any direction', duration: 4,
        narration: 'For the purple unit direction <span class="c-w">u</span> the second slope is the <b>curvature along u</b>: uᵀHu, the quadratic form evaluated on u. Rotate u: the bar peaks at 8 on the steep gold axis and bottoms out at 2 on the gentle one. Every direction’s curvature sits between the two eigenvalues.',
        setup(s) { s.eigGrow = 1; s.eigLevel = 4; s.showProbe = true; s.gauge = true; s.inset = 'u'; s.panel = 'curv'; s.probe = 0; },
        at(s, t) { s.probe = 360 * seg(t, 0, 1, 'linear'); } },

      { title: '✕ Steep is not the same as curved', duration: 4.5, misconception: true,
        narration: '“A big gradient means a sharp bend.” No. Far out along the gentle axis at A the yellow gradient is long but the bend is still λ = 2; near the bottom on the steep axis at B the gradient is short and the bend is 8. For a quadratic, H is the same at every point while ∇f changes everywhere.',
        setup(s) { s.showProbe = false; s.gauge = false; s.probe = 0; s.inset = null; s.eigLevel = null; s.eigGrow = 0; s.mis = 0; s.misCross = false; s.panel = 'mis'; },
        at(s, t) { s.mis = seg(t, 0.05, 0.45, 'out'); s.misCross = t > 0.7; } },

      { title: 'A saddle', duration: 4,
        narration: 'Flip one eigenvalue’s sign: H = [[2, 0], [0, −2]]. The surface rises along the green ' + ihat + ' and falls along the red ' + jhat + '; contours become hyperbolas. A negative eigenvalue is a direction where the slope decreases as you go — a <b>saddle</b>, as in p4-06.',
        setup(s) { s.mis = 0; s.misCross = false; s.view = '3d'; s.cam = cam3({ yaw: -1.75 }); s.surfA = 1; s.showEig = true; s.eigGrow = 1; s.eigLevel = null; s.levels = LVS.slice(); s._H = LA.clone(s.H); s.panel = 'saddle'; },
        at(s, t) { s.H = lerpM(s._H, HS, seg(t, 0.05, 0.75, 'inOut')); LA.cam.tween(s, { yaw: -1.75 }, { yaw: -1.3 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Why ML cares: the safe step', duration: 4.5, mlTie: true,
        narration: 'Near a minimum, L(w) ≈ L(w*) + ½(w − w*)ᵀH(w − w*). Along the steepest bend λmax a step η overshoots when η > 2/λmax; next lesson shows it. Real Hessians are huge; here is one 6×6 as a heatmap. The picture stops at 3. The formula does not.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.showEig = false; s.ml = 0.001; s.panel = 'ml'; },
        at(s, t) { s.ml = 0.001 + 0.999 * seg(t, 0, 1, 'linear'); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Keep three pictures. Slopes of slopes fill a symmetric 2×2 box. A bowl is ½xᵀHx, so H is its shape. The eigenvectors of H are the directions of sharpest and gentlest bend, and the sharpest one limits your step.',
        cards: ['<b>Slopes of slopes, symmetric box.</b> H = [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]]; the mixed partials are equal.', '<b>Bowl = ½xᵀHx.</b> Near a minimum every smooth loss is this quadratic form; H is its shape.', '<b>Eigenvectors = sharpest and gentlest bend.</b> uᵀHu lies between λmin and λmax; λmax limits the step.'],
        setup(s) { s.ml = 0; s.recap = true; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Edit the symmetric <b>H</b> (the lower-left entry mirrors the upper-right), drag the yellow point <span class="c-v">p</span> or the tip of the purple probe <span class="c-w">u</span>.</p>',
      setup(s) { s.view = '2d'; s.cam = cam2(); s.H = LA.clone(HR); s._Hprev = LA.clone(HR); s.levels = LVC.slice(); s.p = [1.2, -0.6]; s.probe = 30; s.showP = true; s.showGrad = true; s.showProbe = true; s.gauge = true; s.showEig = true; s.eigGrow = 1; s.inset = 'u'; s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'H', label: 'H', colors: 'cols', step: 0.5, min: -10, max: 10 },
        { type: 'slider', bind: 'probe', min: 0, max: 360, step: 1, label: 'Probe direction θ', color: 'w', fmt: x => fmt(x, 0) + '°' },
        { type: 'toggle', bind: 'showEig', label: 'Eigen-axes (gold)' },
        { type: 'preset', label: 'Hessian', items: [
          { label: 'diag(2, 8)', apply(s) { s.H = LA.clone(H0); s._Hprev = LA.clone(H0); } },
          { label: 'Rotated', apply(s) { s.H = LA.clone(HR); s._Hprev = LA.clone(HR); } },
          { label: 'Saddle', apply(s) { s.H = LA.clone(HS); s._Hprev = LA.clone(HS); } },
          { label: 'Round I', apply(s) { s.H = LA.clone(HI); s._Hprev = LA.clone(HI); } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2D map', apply(s) { s.view = '2d'; s.cam = cam2(); } },
          { label: '3D surface', apply(s) { s.view = '3d'; s.cam = cam3({ yaw: -1.5 }); } },
        ] },
      ],
    },

    checks: [
      { kind: 'entry',
        prompt: 'Type the Hessian of f(x, y) = 3x² + 2xy + y². The contours of ½xᵀHx for your H are drawn live.',
        bind: 'guessH', answer: [[6, 2], [2, 2]], tol: LA.TOL.entry, label: 'H', colors: 'cols',
        hint: '∂²f/∂x² = 6; the mixed partial is the coefficient of xy, and it appears twice.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.guessH = [[0, 0], [0, 0]]; s.guessMode = true; s.showP = false; s.showEig = false; s.inset = null; s.panel = 'guess'; s.levels = [0.5, 1, 2, 4, 8]; } },
      { kind: 'manipulate',
        prompt: 'Drag the tip of the purple probe <span class="c-w">u</span> to the direction of greatest curvature of H = [[5, 3], [3, 5]].',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.H = LA.clone(HR); s._Hprev = LA.clone(HR); s.levels = LVR.slice(); s.probe = 90; s.showProbe = true; s.gauge = true; s.inset = 'u'; s.showP = false; s.showEig = false; s.panel = 'curv'; },
        judge(s) { const u = unit(s.probe); const a = LA.judge.angleDeg(u, Q1, 0, LA.TOL.angleDeg), b = LA.judge.angleDeg(u, Q1, 180, LA.TOL.angleDeg); const r = a.dist < b.dist ? a : b; return { ok: r.ok, progress: Math.max(0, 1 - r.dist / 90) }; },
        hint: 'Watch the bar — you want it at 8.',
        solution(s) { s.probe = 45; } },
      { kind: 'manipulate',
        prompt: 'Drag the gradient tips at ' + ihat + ' and ' + jhat + ' (they are the columns of H) so the surface becomes a saddle whose downhill direction is ' + jhat + '.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.H = LA.clone(H0); s._Hprev = LA.clone(H0); s.levels = LVC.slice(); s.colHandles = true; s.showEig = true; s.eigGrow = 0; s.showP = false; s.showProbe = false; s.gauge = false; s.inset = null; s.panel = 'saddle'; },
        judge(s) {
          const ev = eigVecs(s.H), lmin = ev.values[1], v = ev.vecs[1];
          const neg = lmin < -LA.TOL.det, ang = Math.acos(Math.min(1, Math.abs(v[1]) / (Math.hypot(v[0], v[1]) || 1))) * 180 / PI;
          return { ok: neg && ang < LA.TOL.angleDeg, progress: (neg ? 0.6 : 0.6 * Math.max(0, 1 - Math.max(0, lmin) / 8)) + (neg ? 0.4 * Math.max(0, 1 - ang / 45) : 0), hint: neg && ang >= LA.TOL.angleDeg ? 'It is a saddle — now keep the off-diagonal at zero so the downhill axis is exactly ' + jhat + '.' : undefined };
        },
        hint: 'Make the bottom-right entry negative (drag the red tip down past ' + jhat + ') and keep the off-diagonals at zero.',
        solution(s) { s.H = LA.clone(HS); s._Hprev = LA.clone(HS); } },
    ],
  });

  function render0(g, s, ctx) {
    if (s.recap) { renderRecap(g, s); return; }
    if (ctx.mode !== 'learn' && !s.guessMode) mirrorH(s);
    if (s.ml > 0) { renderML(g, s); return; }
    if (s.view === '3d') render3d(g, s, ctx); else render2d(g, s, ctx);
  }

  /** Keep H symmetric in Play/Check: whichever off-diagonal entry the learner edited is mirrored. */
  function mirrorH(s) {
    const H = s.H; if (!H || !H[0] || !H[1]) return;
    if (Math.abs(H[0][1] - H[1][0]) > 1e-12) {
      const pv = s._Hprev;
      if (pv && Math.abs(pv[1][0] - H[1][0]) > 1e-12 && Math.abs(pv[0][1] - H[0][1]) < 1e-12) H[0][1] = H[1][0]; else H[1][0] = H[0][1];
    }
    if (!s._Hprev || s._Hprev[0][0] !== H[0][0] || s._Hprev[0][1] !== H[0][1] || s._Hprev[1][0] !== H[1][0] || s._Hprev[1][1] !== H[1][1]) s._Hprev = [[H[0][0], H[0][1]], [H[1][0], H[1][1]]];
  }

  /** Inset rectangle (bottom-right), shared by the slice charts and the flying numbers. */
  function insetRect(size) { const w = Math.max(190, Math.round(size.w * 0.32)), h = 112; return { x: size.w - w - 12, y: size.h - h - 28, w, h }; }

  /** Gold eigen-axes from the origin: length = semi-axis of the level-c ellipse (or a fixed length). */
  function drawEigAxes(g, s, grow, level) {
    const ev = eigVecs(s.H);
    ev.vecs.forEach((v, k) => {
      const lam = ev.values[k];
      let len = 1.6;
      if (level && lam > 1e-6) len = Math.sqrt(2 * level / lam);
      const tip = N.scale(v, len * grow), col = lam < -1e-6 ? goldNeg() : 'gold';
      if (g.is3d) dash3(g, [-v[0] * 3.2, -v[1] * 3.2, 0], [v[0] * 3.2, v[1] * 3.2, 0], { color: col, width: 1, alpha: 0.5 });
      else g.lineThrough([0, 0], v, { len: 5, color: col, width: 1, alpha: 0.45, dashed: true });
      if (grow > 0.01) g.vector(g.is3d ? [0, 0, 0] : [0, 0], g.is3d ? [tip[0], tip[1], 0] : tip, { color: col, width: 3.2, label: 'q' + (k === 0 ? '₁' : '₂') + ', λ = ' + fmt(lam), labelAt: 1, labelSize: 12.5 });
      else if (!g.is3d && s.colHandles) {   // check 3: the live eigen-axes are dashed lines only — name them so the learner can read the feedback
        const e = Math.abs(v[0]) > 0.2 ? v : (v[1] < 0 ? N.scale(v, -1) : v), at = N.scale(e, -2.9);
        g.label('λ = ' + fmt(lam) + (lam < -1e-6 ? ' (downhill axis)' : ''), at, { size: 12, color: lam < -1e-6 ? 'neg' : 'gold', font: 'ui' });
      }
    });
  }

  /* ---------------- 2D map ---------------- */
  function render2d(g, s, ctx) {
    const H = s.guessMode ? symm(s.guessH) : s.H, p = s.p, gr = gQ(H, p), W = ctx.size.w, Hh = ctx.size.h;
    g.ghostGrid();
    g.axes({ names: s.axisNames });
    // check 1 draws only the learner's H (no faint true contours: they would let the answer be matched by eye)
    g.contours(q => fQ(H, q), s.levels, CB, { res: 56, color: 'tick', width: 1.2, alpha: 0.9 });
    if (s.eigLevel && !s.guessMode) g.contours(q => fQ(H, q), [s.eigLevel], CB, { res: 56, color: 'gold', width: 2, alpha: 0.9 });
    if (s.showEig && !s.guessMode) drawEigAxes(g, s, s.eigGrow, s.eigLevel);
    // ghost trail of the green arrow (step 3): equal lengths at every height
    if (s.trail) for (let y = -1; y <= p[1] + 1e-9; y += 0.5) { const q = [p[0], y], gx = gQ(H, q)[0]; g.vector(q, [q[0] + SC * gx, y], { color: 'i', width: 2, alpha: 0.28, head: 8 }); }
    if (s.arrX > 0.01 && Math.abs(gr[0]) > 1e-6) g.vector(p, [p[0] + SC * gr[0] * s.arrX, p[1]], { color: 'i', width: 3, label: '∂f/∂x = ' + fmt(gr[0]), labelAt: 0.5, labelSize: 12.5, dx: 0, dy: 15 });
    if (s.arrY > 0.01 && Math.abs(gr[1]) > 1e-6) g.vector(p, [p[0], p[1] + SC * gr[1] * s.arrY], { color: 'j', width: 3, alpha: s.alphaY, label: s.alphaY > 0.5 ? '∂f/∂y = ' + fmt(gr[1]) : undefined, labelAt: 0.5, labelSize: 12.5, dx: 50, dy: 0 });
    if (s.trail) g.label('unchanged: ∂f/∂x = ' + fmt(gr[0]), [p[0] + SC * gr[0] + 0.15, p[1] + 0.32], { anchor: 'left', size: 12, color: 'muted', font: 'ui' });
    if (s.showGrad && N.norm(gr) > 1e-6) g.vector(p, N.add(p, N.scale(gr, SC)), { color: 'v', width: 3, label: '∇f = Hp', labelAt: 1, labelSize: 12.5 });
    if (s.mis > 0.01) misconception(g, s, ctx);
    if (s.colHandles) colHandles(g, s);
    if (s.showProbe) {
      const u = unit(s.probe);
      g.vector([0, 0], u, { color: 'w', width: 3.2, label: 'u', labelAt: 1 });
      g.handle('u', u, (q, st) => { st.probe = (Math.atan2(q[1], q[0]) * 180 / PI + 360) % 360; }, { color: 'w' });
    }
    if (s.gauge) curvGauge(g, s, ctx.size);
    if (s.inset) sliceInset(g, s, ctx.size);
    if (s.box >= 0) hessBox(g, s, ctx);
    if (s.showP && !s.guessMode && !s.colHandles) {
      g.point(p, { r: 6, color: 'v', label: ctx.mode !== 'learn' ? 'p' : undefined, dx: -13, dy: 12 });
      g.handle('p', p, (q, st) => { st.p = clampP(q); }, { color: 'v' });
    }
  }

  /** Check 3: the columns of H are the gradients at î and ĵ; drag their tips (H stays symmetric). */
  function colHandles(g, s) {
    const H = s.H, bi = [1, 0], bj = [0, 1];
    const ti = [1 + SCC * H[0][0], SCC * H[1][0]], tj = [SCC * H[0][1], 1 + SCC * H[1][1]];
    g.point(bi, { r: 4, color: 'i', label: 'î', dx: 0, dy: 14 }); g.point(bj, { r: 4, color: 'j', label: 'ĵ', dx: -14, dy: 0 });
    g.vector(bi, ti, { color: 'i', width: 3, label: '∇f(î) = Hî', labelAt: 1, labelSize: 12 });
    g.vector(bj, tj, { color: 'j', width: 3, label: '∇f(ĵ) = Hĵ', labelAt: 1, labelSize: 12 });
    g.handle('ci', ti, (q, st) => { const a = (q[0] - 1) / SCC, b = q[1] / SCC; st.H[0][0] = a; st.H[1][0] = b; st.H[0][1] = b; st._Hprev = LA.clone(st.H); }, { color: 'i' });
    g.handle('cj', tj, (q, st) => { const c = q[0] / SCC, d = (q[1] - 1) / SCC; st.H[0][1] = c; st.H[1][0] = c; st.H[1][1] = d; st._Hprev = LA.clone(st.H); }, { color: 'j' });
  }

  /** Vertical gauge of uᵀHu (screen px, bottom-left), with gold ticks at the eigenvalues. */
  function curvGauge(g, s, size) {
    const ev = eigVecs(s.H), l1 = ev.values[0], l2 = ev.values[1], val = curvU(s.H, unit(s.probe));
    const lo = Math.min(0, l2) - 1, hi = Math.max(l1, 1) + 1;
    const x = 34, w = 28, h = 150, y = size.h - h - 46;
    const Y = v => y + h - (v - lo) / (hi - lo) * h, Wd = (sx, sy) => g.P.toWorld(sx, sy);
    g.polygon([Wd(x - 8, y - 6), Wd(x + w + 92, y - 6), Wd(x + w + 92, y + h + 6), Wd(x - 8, y + h + 6)], { fill: 'panel', alpha: 0.85, stroke: 'line', width: 1 });
    const y0 = Y(0), yv = Y(val);
    if (Math.abs(yv - y0) > 0.5) g.polygon([Wd(x, y0), Wd(x + w, y0), Wd(x + w, yv), Wd(x, yv)], { fill: val >= 0 ? 'w' : 'neg', alpha: 0.9 });
    g.line(Wd(x - 6, y0), Wd(x + w + 6, y0), { color: 'hud', width: 1.2 });
    g.label('0', [x + w + 10, y0], { screen: true, anchor: 'left', size: 11, color: 'hud', font: 'ui', bg: false });
    [[l1, 'λ₁'], [l2, 'λ₂']].forEach(([lv, nm]) => { const yy = Y(lv); g.line(Wd(x - 6, yy), Wd(x + w + 6, yy), { color: lv < 0 ? 'neg' : 'gold', width: 2 }); g.label(nm + ' = ' + fmt(lv), [x + w + 10, yy], { screen: true, anchor: 'left', size: 11.5, color: lv < 0 ? 'neg' : 'gold', font: 'ui', bg: false }); });
    g.label(fmt(val), [x + w / 2, val >= 0 ? yv - 9 : yv + 9], { screen: true, size: 11.5, color: 'w', font: 'ui', bg: false });
    g.label('curvature uᵀHu', [x - 8, y - 18], { screen: true, anchor: 'left', size: 12, color: 'muted', font: 'ui' });
  }

  /** Inset slice charts: 'x' slice through p, 'xy' both, 'box' both + second-derivative labels, 'eig' along q₁/q₂, 'u' along u. */
  function sliceInset(g, s, size) {
    const r = insetRect(size), H = s.H, p = s.p, gr = gQ(H, p), mode = s.inset;
    const xr = [-3, 3], yr = [-2, 10];
    const mk = (fn, color, dash, lo, hi) => { const pts = []; lo = lo === undefined ? xr[0] : lo; hi = hi === undefined ? xr[1] : hi; for (let i = 0; i <= 72; i++) { const v = lo + (hi - lo) * i / 72, f = fn(v); if (f <= yr[1] && f >= yr[0]) pts.push([v, f]); } return { points: pts, color, width: 1.8, dash }; };
    const X = v => r.x + (v - xr[0]) / (xr[1] - xr[0]) * r.w, Y = v => r.y + r.h - (v - yr[0]) / (yr[1] - yr[0]) * r.h;
    const series = []; let title = '';
    if (mode === 'x') {
      series.push(mk(v => fQ(H, [v, p[1]]), 'i'));
      const f0 = fQ(H, p), m = gr[0];
      series.push({ points: [[p[0] - 0.9, f0 - 0.9 * m], [p[0] + 0.9, f0 + 0.9 * m]], color: 'i', width: 2.6 });
      title = 'slice f(x, ' + fmt(p[1]) + '): slope = 2x';
      g.lineChart(series, { x: r.x, y: r.y, w: r.w, h: r.h, xRange: xr, yRange: yr, title, xLabel: 'x' });
      if (f0 <= yr[1]) g.label('●', [X(p[0]), Y(f0)], { screen: true, color: 'v', bg: false, size: 13 });
      g.label('slope ' + fmt(m), [X(p[0]) + (m >= 0 ? -8 : 8), Y(f0) + 16], { screen: true, anchor: m >= 0 ? 'right' : 'left', size: 11.5, color: 'i', font: 'ui' });
      return;
    }
    if (mode === 'xy' || mode === 'box') {
      series.push(mk(v => fQ(H, [v, 0]), 'i')); series.push(mk(v => fQ(H, [0, v]), 'j'));
      title = 'slices: x (green), y (red)';
      g.lineChart(series, { x: r.x, y: r.y, w: r.w, h: r.h, xRange: xr, yRange: yr, title, xLabel: 'x or y' });
      if (mode === 'box' && s.box < 0.05) { g.label('∂²f/∂x² = 2', [r.x + 8, r.y + 16], { screen: true, anchor: 'left', size: 12, color: 'i', font: 'ui' }); g.label('∂²f/∂y² = 8', [r.x + r.w - 8, r.y + 16], { screen: true, anchor: 'right', size: 12, color: 'j', font: 'ui' }); }
      return;
    }
    if (mode === 'eig') {
      const ev = eigVecs(H);
      series.push(mk(v => 0.5 * ev.values[0] * v * v, 'gold')); series.push(mk(v => 0.5 * ev.values[1] * v * v, 'gold', [5, 4]));
      g.lineChart(series, { x: r.x, y: r.y, w: r.w, h: r.h, xRange: xr, yRange: yr, title: 'slices along q₁ (solid) and q₂ (dashed)', xLabel: 'distance along the axis' });
      g.label('λ₁ = ' + fmt(ev.values[0]), [X(0.9), Y(7)], { screen: true, anchor: 'right', size: 11.5, color: 'gold', font: 'ui' });
      g.label('λ₂ = ' + fmt(ev.values[1]), [X(2.9), Y(5.5)], { screen: true, anchor: 'right', size: 11.5, color: 'gold', font: 'ui' });
      return;
    }
    if (mode === 'u') {
      const k = curvU(H, unit(s.probe));
      series.push(mk(v => 0.5 * k * v * v, 'w'));
      g.lineChart(series, { x: r.x, y: r.y, w: r.w, h: r.h, xRange: xr, yRange: yr, title: 'slice along u: ½(uᵀHu)t²', xLabel: 't (distance along u)' });
      g.label('uᵀHu = ' + fmt(k), [r.x + r.w - 8, r.y + 16], { screen: true, anchor: 'right', size: 12, color: 'w', font: 'ui' });
    }
  }

  /** Step 4: the four second derivatives fly from the inset into an on-canvas 2×2 box; the diagonal pulses. */
  function hessBox(g, s, ctx) {
    const W = (sx, sy) => g.P.toWorld(sx, sy), r = insetRect(ctx.size), H = s.H;
    const fly = LA.clamp01(s.box), pulse = s.box > 1 ? Math.sin(PI * (s.box - 1) * 2) : 0;
    const bx = 34, by = 70, cw = 56, ch = 40;
    const cell = (i, j) => [bx + 80 + j * cw, by + 22 + i * ch];
    const ps = g.P.toScreen(s.p);
    const from = { '00': [r.x + 40, r.y + 16], '11': [r.x + r.w - 40, r.y + 16], '01': [ps[0] + 70, ps[1] - 30], '10': [ps[0] + 70, ps[1] - 30] };
    g.polygon([W(bx - 6, by - 10), W(bx + 70 + 2 * cw + 14, by - 10), W(bx + 70 + 2 * cw + 14, by + 2 * ch + 20), W(bx - 6, by + 2 * ch + 20)], { fill: 'panel', alpha: 0.85 * Math.min(1, fly * 3), stroke: 'line', width: 1 });
    g.label('H =', [bx + 26, by + 22 + ch / 2], { screen: true, size: 18, color: 'text', font: 'math', bg: false, alpha: Math.min(1, fly * 3) });
    const L = bx + 62, R = bx + 62 + 2 * cw + 10, T = by + 2, B = by + 2 * ch + 4;
    [[L, 1], [R, -1]].forEach(([xx, d]) => g.polyline([W(xx + 6 * d, T), W(xx, T), W(xx, B), W(xx + 6 * d, B)], { color: 'text', width: 1.6, alpha: Math.min(1, fly * 3) }));
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      const key = '' + i + j, a = from[key], b = cell(i, j), q = lerpV(a, b, fly);
      const diag = i === j, col = diag ? (i === 0 ? 'i' : 'j') : 'text';
      g.label(fmt(H[i][j]), q, { screen: true, size: 19 + (diag ? 5 * Math.max(0, pulse) : 0), color: col, font: 'math', bold: diag, bg: false });
    }
    if (fly > 0.95) {
      const a = cell(0, 1), b = cell(1, 0), m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      g.vector(W(m[0], m[1]), W(a[0] - 8, a[1] + 8), { color: 'muted', width: 1.4, head: 7 });
      g.vector(W(m[0], m[1]), W(b[0] + 8, b[1] - 8), { color: 'muted', width: 1.4, head: 7 });
      g.label('equal: symmetric', [bx + 70 + cw, B + 26], { screen: true, size: 11.5, color: 'muted', font: 'ui', bg: false });
    }
  }

  /** Step 9: a long gradient with a gentle bend (A) vs a short gradient with a sharp bend (B). */
  function misconception(g, s, ctx) {
    const H = s.H, a = s.mis, A = N.scale(Q2, 2), B = N.scale(Q1, 0.3);
    const gA = gQ(H, A), gB = gQ(H, B);
    g.point(A, { r: 5.5, color: 'v', label: 'A', dx: -14, dy: 10 });
    g.point(B, { r: 5.5, color: 'v', label: 'B', dx: -12, dy: 12 });
    if (a > 0.01) {
      g.vector(A, N.add(A, N.scale(gA, SC * a)), { color: 'v', width: 3.2, label: '‖∇f‖ = ' + fmt(N.norm(gA)), labelAt: 0.55, labelSize: 12.5, dx: -44, dy: 4 });
      g.vector(B, N.add(B, N.scale(gB, SC * a)), { color: 'v', width: 3.2, label: '‖∇f‖ = ' + fmt(N.norm(gB)), labelAt: 1, labelSize: 12.5 });
    }
    // two mini charts with the same axes: slice along the gradient direction, minus nothing — plus its tangent
    const W = ctx.size.w, Hh = ctx.size.h, cw = Math.min(200, (W - 60) / 2), chh = 78, y = Hh - chh - 26;
    const chart = (x, P, gP, lam, name) => {
      const n = N.norm(gP), xr = [-0.6, 0.6], yr = [-2.6, 2.6];
      const curve = [], tan = [];
      for (let i = 0; i <= 48; i++) { const t = xr[0] + (xr[1] - xr[0]) * i / 48; curve.push([t, n * t + 0.5 * lam * t * t]); tan.push([t, n * t]); }
      g.lineChart([{ points: tan, color: 'muted', width: 1.2, dash: [4, 4] }, { points: curve, color: 'v', width: 2 }], { x, y, w: cw, h: chh, xRange: xr, yRange: yr, title: 'at ' + name + ': slope ' + fmt(n) + ', bend λ = ' + fmt(lam), xLabel: 'step along ∇f' });
    };
    if (a > 0.3) { chart(16, A, gA, 2, 'A'); chart(W - cw - 16, B, gB, 8, 'B'); }
    const lp = [-3.1, 3.3];
    g.label('“steep ⇒ curved”', lp, { anchor: 'left', size: 13.5, color: 'muted', font: 'ui' });
    if (s.misCross) g.marker.cross(lp, { size: 22, dx: -16 });
  }

  /* ---------------- 3D surface ---------------- */
  function drawSurface(g, H, alpha) {
    g.paramSurface((u, v) => {
      const d = [Math.cos(v), Math.sin(v)], k = curvU(H, d);
      const rm = k > 1e-6 ? Math.min(RMAX, Math.sqrt(2 * CAP / k)) : RMAX;
      const x = u * rm * d[0], y = u * rm * d[1];
      return [x, y, ZS * fQ(H, [x, y])];
    }, { u0: 0, u1: 1, v0: 0, v1: 2 * PI }, { res: 24, mode: 'both', color: 'grid', alpha: 0.26 * alpha, wireColor: 'gridAxis', wireAlpha: 0.45 * alpha });
  }
  function render3d(g, s, ctx) {
    const H = s.H;
    g.cage({ range: 3, walls: false, alpha: 0.6 });
    g.contours(q => fQ(H, q), s.levels, CB, { res: 44, color: 'tick', width: 1.1, alpha: 0.85, z: 0 });
    if (s.eigLevel) g.contours(q => fQ(H, q), [s.eigLevel], CB, { res: 44, color: 'gold', width: 2.2, z: 0 });
    drawSurface(g, H, s.surfA);
    g.axes({ length: 3.4, names: ['x', 'y', 'f'] });
    if (s.showEig) drawEigAxes(g, s, s.eigGrow, s.eigLevel);
    if (s.showP && ctx.mode !== 'learn') {
      const p = s.p, P3 = [p[0], p[1], 0];
      g.point([p[0], p[1], ZS * fQ(H, p)], { r: 4, color: 'v' }); dash3(g, P3, [p[0], p[1], ZS * fQ(H, p)], { color: 'v', width: 1.2 });
      g.point(P3, { r: 6, color: 'v', label: 'p', dx: -13, dy: 12 });
      g.handle('p', P3, (q, st) => { st.p = clampP(q); }, { color: 'v', plane: 'xy' });
    }
  }

  /* ---------------- mlTie: the safe step along the steepest bend + a 6×6 Hessian heatmap ---------------- */
  function renderML(g, s) {
    const t = s.ml, lam = 8;
    const O = [-1.75, -1.45], sx = 1.05, sy = 0.5;
    const P = (a, L) => [O[0] + sx * a, O[1] + sy * L];
    // parabola L(a) = ½·8·a²
    const pts = []; for (let i = 0; i <= 80; i++) { const a = -1.5 + 3 * i / 80; pts.push(P(a, 0.5 * lam * a * a)); }
    g.line(P(-1.55, 0), P(1.55, 0), { color: 'gridAxis', width: 1.2 });
    g.polyline(pts, { color: 'gold', width: 2.4 });
    g.label('slice along q₁: L = ½·λmax·a², λmax = 8', P(0, -0.95), { size: 12.5, color: 'gold', font: 'ui' });
    g.label('one step: a ← a − η·λmax·a', P(0, -1.75), { size: 12, color: 'b', font: 'ui' });
    g.point(P(0, 0), { r: 5, color: 'gold', label: 'w*', dx: 0, dy: 14 });
    const a0 = 1, start = P(a0, 4);
    g.point(start, { r: 6, color: 'b', label: 'a = 1', dx: 26, dy: -4 });
    const step = (eta, u, good) => {
      if (u <= 0.01) return;
      const a1 = a0 - eta * lam * a0, end = P(a1, 0.5 * lam * a1 * a1), mid = LA.lerpV(start, end, u);
      g.vector(start, mid, { color: 'b', width: 2.8, head: 10 });
      if (u > 0.98) {
        g.point(end, { r: 5.5, color: 'b' });
        g.label('η = ' + fmt(eta) + ' → a = ' + fmt(a1) + (good ? ' (closer)' : ' (farther out)'), end, { anchor: 'left', dx: 12, dy: good ? 16 : 18, size: 12, color: good ? 'ok' : 'bad', font: 'ui' });
        if (good) g.marker.check(end, { dx: -16, dy: -14, size: 18 }); else g.marker.cross(end, { dx: 18, dy: -14, size: 20 });
      }
    };
    step(0.1, seg(t, 0.1, 0.4, 'inOut'), true);
    step(0.3, seg(t, 0.45, 0.75, 'inOut'), false);
    // η axis with the red limit
    const ey = -3.0, EX = e => -3.2 + 8 * e;
    g.line([EX(0), ey], [EX(0.37), ey], { color: 'hud', width: 1.4 });
    [0, 0.1, 0.2, 0.3].forEach(e => { g.line([EX(e), ey - 0.07], [EX(e), ey + 0.07], { color: 'hud', width: 1.2 }); g.label(fmt(e), [EX(e), ey - 0.25], { size: 11, color: 'tick', font: 'ui', bg: false }); });
    g.label('η', [EX(0.37) + 0.15, ey], { size: 14, color: 'hud', font: 'math' });
    const red = seg(t, 0.75, 0.95);
    if (red > 0.01) { g.line([EX(0.25), ey - 0.45 * red], [EX(0.25), ey + 0.45 * red], { color: 'bad', width: 2.4 }); g.label('2/λmax = 0.25', [EX(0.25), ey - 0.62], { size: 12, color: 'bad', font: 'ui', alpha: red }); }
    if (t > 0.4) g.point([EX(0.1), ey], { r: 5, color: 'ok' });
    if (t > 0.75) g.point([EX(0.3), ey], { r: 5, color: 'bad' });
    // 6×6 Hessian heatmap (synthetic, seeded)
    const fade = seg(t, 0.2, 0.6);
    if (fade > 0.01) {
      const b = { x0: 0.75, x1: 3.15, y0: -1.1, y1: 1.3 }, M = s.heat6, n = 6, cw = (b.x1 - b.x0) / n;
      const m = Math.max(...M.map(r => Math.max(...r.map(Math.abs))));
      g.heat2d(M, b, { range: [-m, m], ramp: x => LA.color.diverge(2 * x - 1), smooth: false, alpha: 0.95 * fade });
      for (let i = 0; i <= n; i++) { g.line([b.x0 + i * cw, b.y0], [b.x0 + i * cw, b.y1], { color: 'line', width: 1, alpha: fade }); g.line([b.x0, b.y0 + i * cw], [b.x1, b.y0 + i * cw], { color: 'line', width: 1, alpha: fade }); }
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) g.label(LA.fmt.fixed(M[i][j], 1), [b.x0 + (j + 0.5) * cw, b.y1 - (i + 0.5) * cw], { size: 10, color: 'text', bg: false, font: 'ui', alpha: fade });
      g.label('a 6×6 Hessian (synthetic, seeded)', [(b.x0 + b.x1) / 2, b.y1 + 0.3], { size: 12, color: 'hud', font: 'ui', alpha: fade });
      g.label('symmetric: mirror across the diagonal', [(b.x0 + b.x1) / 2, b.y0 - 0.3], { size: 11, color: 'muted', font: 'ui', alpha: fade });
    }
    if (t > 0.5) g.label('The picture stops at 3. The formula does not.', [0, 3.55], { size: 13.5, color: 'hud', font: 'ui', alpha: seg(t, 0.5, 0.7) });
  }

  /* ---------------- recap: three mini-cards ---------------- */
  function renderRecap(g, s) {
    const card = (i, title, view, cam, fn) => {
      g.viewport({ x: 0.04, y: 0.02 + i * 0.33, w: 0.92, h: 0.31 }, { view, cam, bg: 'panel' }, (gg, r) => { fn(gg, r); gg.label(title, [r.x + 12, r.y + 16], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true, bg: false }); });
    };
    card(0, '1 · Slopes of slopes, symmetric box', '2d', { center: [0.3, 1.35], halfH: 1.75 }, gg => {
      const pa = [], pb = [];
      for (let i = 0; i <= 60; i++) { const v = -1.6 + 3.2 * i / 60; if (v * v <= 2.4) pa.push([v - 1.4, v * v * 0.9]); if (4 * v * v <= 2.4) pb.push([v - 1.4, 4 * v * v * 0.9]); }
      gg.polyline(pa, { color: 'i', width: 2.2 }); gg.polyline(pb, { color: 'j', width: 2.2 });
      gg.label('∂²f/∂x² = 2', [-3.0, 2.45], { size: 12, color: 'i', font: 'ui' }); gg.label('∂²f/∂y² = 8', [-1.4, 2.45], { size: 12, color: 'j', font: 'ui' });
      gg.label('H = [[2, 0], [0, 8]]', [1.9, 1.3], { size: 17, color: 'text', font: 'math' });
      gg.label('mixed partials equal ⇒ symmetric', [1.9, 0.75], { size: 11.5, color: 'muted', font: 'ui' });
    });
    card(1, '2 · Bowl = ½xᵀHx', '3d', cam3({ dist: 6.8, yaw: -1.6, pitch: 0.45, target: [0, 0, 0.7] }), gg => {
      gg.contours(q => fQ(H0, q), LV0, CB, { res: 36, color: 'tick', width: 1, z: 0 });
      gg.contours(q => fQ(H0, q), [2], CB, { res: 36, color: 'gold', width: 2, z: 0 });
      drawSurface(gg, H0, 1);
    });
    card(2, '3 · Eigenvectors = sharpest and gentlest bend', '2d', { center: [0.4, 0], halfH: 1.9 }, gg => {
      gg.contours(q => fQ(HR, q), [1, 2, 4, 8], CB, { res: 56, color: 'tick', width: 1.1 });
      gg.contours(q => fQ(HR, q), [4], CB, { res: 56, color: 'gold', width: 2 });
      gg.vector([0, 0], N.scale(Q1, 1), { color: 'gold', width: 3, label: 'λ = 8 (sharp)', labelAt: 1, labelSize: 12 });
      gg.vector([0, 0], N.scale(Q2, 2), { color: 'gold', width: 3, label: 'λ = 2 (gentle)', labelAt: 1, labelSize: 12 });
    });
  }
})();
