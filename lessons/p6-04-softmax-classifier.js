/* p6-04 — Softmax classifier geometry. Regions: g.heat2d of argmax with a categorical ramp. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, C = LA.color, lerp = LA.lerp, lerpV = LA.lerpV, clamp01 = LA.clamp01;
  const W_DEF = [[1.0, 1.0], [-1.2, 0.3], [0.3, -1.1]];
  const W_BAD = [[0.3, -0.8], [-0.5, -0.6], [0.9, 0.4]];
  const CLS = ['data', 'dataB', 'b'], NAMES = ['A', 'B', 'C'];
  const BOUNDS = { x0: -3.4, x1: 3.4, y0: -3.4, y1: 3.4 };
  const ETA = 0.5;
  const catRamp = t => t < 0.25 ? C.data : (t < 0.75 ? C.dataB : C.b);

  // ---------- data + deterministic training, computed once ----------
  let CACHE = null;
  function heavy() {
    if (CACHE) return CACHE;
    const bl = LA.rng.blobs(604, 120, { centers: [[1.5, 1.5], [-1.5, 0.5], [0.5, -1.5]], spread: 0.5 });
    const ci = LA.rng.circles(605, 100, {});
    const r2 = v => Math.round(v * 100) / 100;
    const blobs = { X: bl.X.map(p => p.map(r2)), y: bl.y };
    const circles = { X: ci.X.map(p => p.map(r2)), y: ci.y };
    const hist = [LA.clone(W_BAD)], grads = [];
    let W = LA.clone(W_BAD);
    for (let k = 0; k < 10; k++) { const G = grad(W, [0, 0, 0], blobs); grads.push(G.map(r => r.map(r2))); W = N.subM(W, N.scaleM(G, ETA)); hist.push(W.map(r => r.map(v => Math.round(v * 1000) / 1000))); }
    let Ws = LA.clone(W);
    for (let k = 0; k < 400; k++) Ws = N.subM(Ws, N.scaleM(grad(Ws, [0, 0, 0], blobs), 1.0));
    Ws = Ws.map(r => r.map(v => Math.round(v * 1000) / 1000));
    const rand = LA.rng(606), Wbig = [];
    for (let i = 0; i < 10; i++) { const row = []; for (let j = 0; j < 16; j++) row.push(Math.round(LA.rng.gauss(rand) * 100) / 100); Wbig.push(row); }
    const feat = []; for (let j = 0; j < 16; j++) feat.push(Math.round(LA.rng.gauss(rand) * 100) / 100);
    // the best straight split of the rings (any direction, any offset; sampled every 2° and every 0.05)
    let mis = { n: [1, 0], c: 0, acc: 0 };
    for (let a = 0; a < 360; a += 2) {
      const n = [Math.cos(a * Math.PI / 180), Math.sin(a * Math.PI / 180)];
      for (let c = -2.5; c <= 2.5; c += 0.05) {
        let ok = 0; circles.X.forEach((x, i) => { if ((n[0] * x[0] + n[1] * x[1] - c >= 0 ? 0 : 1) === circles.y[i]) ok++; });
        if (ok > mis.acc * circles.X.length) mis = { n, c, acc: ok / circles.X.length };
      }
    }
    CACHE = { blobs, circles, hist, grads, Ws, Wbig, feat, mis };
    return CACHE;
  }
  function scoresOf(W, b, x, gain) { return [0, 1, 2].map(c => gain * (W[c][0] * x[0] + W[c][1] * x[1]) + b[c]); }
  function probs(W, b, x, gain, T) { return N.softmax(scoresOf(W, b, x, gain), T); }
  function argmax(v) { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; }
  function grad(W, b, D) {
    const G = [[0, 0], [0, 0], [0, 0]], n = D.X.length;
    D.X.forEach((x, i) => { const p = probs(W, b, x, 1, 1); for (let c = 0; c < 3; c++) { const e = p[c] - (D.y[i] === c ? 1 : 0); G[c][0] += e * x[0] / n; G[c][1] += e * x[1] / n; } });
    return G;
  }
  function accuracy(W, b, D, gain) { let ok = 0; D.X.forEach((x, i) => { if (argmax(scoresOf(W, b, x, gain || 1)) === D.y[i]) ok++; }); return ok / D.X.length; }
  function dataOf(s) { return s.preset === 'circles' ? s.circ : s.blob; }
  function Wcur(s) { if (s.gdK >= 0) { const k = Math.min(9, Math.floor(s.gdK)), f = s.gdK - k; return LA.lerpM(s.hist[k], s.hist[Math.min(10, k + 1)], s.gdK >= 10 ? 1 : f); } return s.W; }
  /** the point where all three scores tie */
  function triplePoint(W, b) {
    const A = [[W[0][0] - W[1][0], W[0][1] - W[1][1]], [W[0][0] - W[2][0], W[0][1] - W[2][1]]];
    if (Math.abs(N.det(A)) < 1e-9) return null;
    return N.solve(A, [b[1] - b[0], b[2] - b[0]]);
  }

  // ---------- screen helpers ----------
  function rect(g, x, y, w, h, fill) { g.strip([0], { x, y, w, h, range: [0, 1], ramp: () => fill }); }
  function txt(g, s, x, y, o) { g.label(s, [x, y], Object.assign({ screen: true, size: 12.5, bg: false, font: 'ui', anchor: 'left', color: 'hud' }, o || {})); }
  function panelBg(g, x, y, w, h) { rect(g, x, y, w, h, C.alpha(C.panel, 0.88)); }

  const cam2 = () => ({ center: [0, 0], halfH: 3.3 });
  const cam3p = () => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.3], dist: 11, yaw: -1.22, pitch: 0.5, fov: 36 });
  const ZS = 0.5;   // score planes drawn at half height

  LA.registerLesson({
    id: 'p6-04-softmax-classifier',
    part: 6,
    title: 'Softmax classifier geometry',
    subtitle: 'Three class arrows tile the plane; softmax puts the scores on a triangle; the gradient rotates the arrows.',
    minutes: 12,
    prereqs: ['p1-09-first-model-neuron', 'p4-04-simplex-softmax', 'p5-01-gradient', 'p5-03-gradient-descent', 'p5-06-chain-rule-backprop', 'p6-01-batches-shapes-tensors'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D features + 3D score planes + the 3-class probability triangle', realSize: 'an ImageNet head is W ∈ ℝ¹⁰⁰⁰ˣ²⁰⁴⁸', transfers: 'scores are dot products, argmax regions are convex, softmax acts on the score vector, ∂L/∂W = (p − y)xᵀ — exact for any d and any number of classes' },
    cantShow: ['more than three classes on the triangle (4 classes need a tetrahedron; 1000 cannot be drawn)', 'the 2048-D feature space; regions there are convex cones we only name', 'the lift that produces the features (p2-12, p6-05)'],
    terms: [
      { term: 'score', def: 'For class c, s_c = w_c·x + b_c (also called a logit): one number per class.' },
      { term: 'class arrow', def: 'The weight vector w_c of class c: a direction in feature space.' },
      { term: 'decision region', def: 'All points where class c has the highest score.' },
      { term: 'decision boundary', def: 'Where two scores tie: a straight line perpendicular to the difference of their class arrows.' },
      { term: 'probability triangle', def: 'The set of non-negative triples that sum to one (the simplex of p4-04).' },
      { term: 'one-hot', def: 'A target vector y with a 1 at the true class and 0 elsewhere.' },
      { term: 'cross-entropy', def: 'The loss −log p_true: small when the true class gets high probability.' },
    ],
    atlas: ['softmax', 'logistic-regression', 'optimization', 'backprop'],
    thumbnail: 3,

    state() {
      const H = heavy();
      return {
        view: '2d', cam: cam2(),
        blob: H.blobs, circ: H.circles, hist: H.hist, grads: H.grads, Wsol: H.Ws, Wbig: H.Wbig, feat: H.feat,
        W: LA.clone(W_DEF), b: [0, 0, 0], probe: [0.8, 0.2], T: 1, gain: 1, preset: 'blobs', gdK: -1,
        scene: 'plane', binary: 0, sheet: 0, arrowsA: 0, showProbe: false, shadows: 0, showBars: false, regionsA: 0, edges: 0,
        diffU: 0, showSimplex: false, planesA: 0, misU: 0, liftB: 0, u: 0, mathMode: 'bin', guessDiff: [1, 0], checkMode: null, dragArrows: false, dragProbe: false,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'ml') { drawML(g, s); return; }
      if (g.is3d) draw3d(g, s); else draw2d(g, s, ctx);
      drawInsets(g, s);
    },

    math(s, tex) {
      const W = Wcur(s), x = s.probe;
      const p = probs(W, s.b, x, s.gain, s.T);
      switch (s.mathMode) {
        case 'bin': return tex.block(tex.eq(tex.sym('p'), tex.op('='), tex.text('σ('), tex.sym('w', { color: 'w' }), tex.op('·'), tex.sym('x'), tex.op('+'), tex.sym('b'), tex.text(')')));
        case 'scores': return tex.block(tex.eq(tex.sym('s'), tex.op('='), tex.sym('W'), tex.sym('x'), tex.op('+'), tex.sym('b'), tex.op('='), tex.mat(W, { colors: (i) => CLS[i] }), tex.vec(x), tex.op('='), tex.vec(scoresOf(W, s.b, x, s.gain), { color: 'gold' })), tex.eq(tex.text('(3×2)(2×1)')));
        case 'argmax': return tex.block(tex.eq(tex.text('class(x) = argmax'), tex.sym('', { sub: 'c' }), tex.sym('s', { sub: 'c' }), tex.text('(x)')));
        case 'edge': return tex.block(tex.eq(tex.text('edge A|B:'), tex.sym('s', { sub: 'A' }), tex.op('='), tex.sym('s', { sub: 'B' })));
        case 'diff': return tex.block(tex.eq(tex.paren(tex.sym('w', { sub: 'A' }) + tex.op('−') + tex.sym('w', { sub: 'B' })), tex.op('·'), tex.sym('x'), tex.op('+'), tex.paren(tex.sym('b', { sub: 'A' }) + tex.op('−') + tex.sym('b', { sub: 'B' })), tex.op('='), tex.num(0)), tex.eq(tex.sym('w', { sub: 'A' }), tex.op('−'), tex.sym('w', { sub: 'B' }), tex.op('='), tex.vec(N.sub(W[0], W[1]), { color: 'w' })));
        case 'soft': return tex.block(tex.eq(tex.sym('p'), tex.op('='), tex.text('softmax(s)'), tex.op('='), tex.rowvec(p, { color: 'gold' })));
        case 'gain': return tex.block(tex.eq(tex.text('softmax(' + LA.fmt(s.gain) + '·s) ='), tex.rowvec(p, { color: 'gold' })), tex.eq(tex.text('regions unchanged')));
        case 'mis': return tex.block(tex.eq(tex.text('argmax of linear scores ⇒ straight edges')));
        case 'gd': { const k = Math.min(10, Math.floor(s.gdK + 1e-9)); return tex.block(tex.eq(tex.text('∂L/∂W = mean'), tex.sym('', { sub: 'n' }), tex.paren(tex.sym('p') + tex.op('−') + tex.sym('y')), tex.T(tex.sym('x'))), tex.eq(tex.sym('W'), tex.op('←'), tex.sym('W'), tex.op('−'), tex.num(ETA), tex.text('∂L/∂W')), tex.eq(tex.text('step ' + k + ' · accuracy '), tex.num(LA.fmt.pct(accuracy(Wcur(s), [0, 0, 0], s.blob)), { color: 'gold' }))); }
        case 'bias': return tex.block(tex.eq(tex.sym('S'), tex.op('='), tex.sym('X̃'), tex.T(tex.sym('W̃')), tex.op('·'), tex.sym('X̃'), tex.op('='), tex.text('[X 1]')), tex.eq(tex.sym('b'), tex.op('='), tex.rowvec(s.b)));
        case 'ml': return tex.block(tex.eq(tex.sym('p'), tex.op('='), tex.text('softmax(Wφ(x) + b)'), tex.op('·'), tex.sym('W'), tex.op('∈'), tex.text('ℝ¹⁰⁰⁰ˣ²⁰⁴⁸')));
        case 'play': return tex.block(tex.eq(tex.sym('W'), tex.op('='), tex.mat(s.W, { colors: (i) => CLS[i] }), tex.sym('b'), tex.op('='), tex.vec(s.b)), tex.eq(tex.sym('p'), tex.op('='), tex.rowvec(p, { color: 'gold' }), tex.text(' accuracy '), tex.num(LA.fmt.pct(accuracy(s.W, s.b, dataOf(s), s.gain)), { color: 'gold' })));
        case 'guess': return tex.block(tex.eq(tex.text('your w_A − w_B ='), tex.vec(s.guessDiff, { color: 'w' })));
        default: return '';
      }
    },

    steps: [
      { title: 'One arrow, one boundary (recall)', duration: 3.5,
        narration: 'From p1-09: a weight arrow <span class="c-w">w</span> and the line perpendicular to it where w·x + b = 0. Points on one side score positive, on the other negative; the sigmoid turns the score into a probability. One moment to remember, then we add classes.',
        setup(s) { Object.assign(s, { view: '2d', cam: cam2(), scene: 'plane', binary: 1, sheet: 0, arrowsA: 0, showProbe: false, shadows: 0, showBars: false, regionsA: 0, edges: 0, diffU: 0, showSimplex: false, planesA: 0, misU: 0, liftB: 0, W: LA.clone(W_DEF), b: [0, 0, 0], gain: 1, T: 1, preset: 'blobs', gdK: -1, mathMode: 'bin', u: 0 }); },
        at(s, t) {
          s.u = t; const on3 = t > 0.45 && t < 0.88;
          s.view = on3 ? '3d' : '2d'; s.sheet = on3 ? 1 : 0;
          if (on3) { s.cam = cam3p(); s.cam.dist = 13; s.cam.yaw = lerp(-1.4, -1.0, seg(t, 0.45, 0.88, 'linear')); } else s.cam = cam2();
        } },

      { title: 'Three classes, three arrows', duration: 4,
        narration: 'Three blobs need three <b>class arrow</b>s w_A, w_B, w_C, one per class, each colored like its blob. A probe point scores against each arrow by its shadow on it (p1-06): three dot products, three <b>score</b>s, three bars. The highest bar names the class.',
        setup(s) { Object.assign(s, { view: '2d', cam: cam2(), binary: 0, sheet: 0, arrowsA: 0, showProbe: true, shadows: 0, showBars: false, mathMode: 'scores', u: 0 }); },
        at(s, t) { s.u = t; s.arrowsA = seg(t, 0.05, 0.35); s.shadows = seg(t, 0.4, 0.75); s.showBars = t > 0.7; } },

      { title: 'Three tilted planes', duration: 4.5,
        narration: 'Each score is a tilted plane over the data plane (p1-09 made taller): height s_c(x) at every point. Three planes, three tilts. Standing at the probe, the tallest plane wins. Orbit to see them cross.',
        setup(s) { Object.assign(s, { view: '3d', shadows: 0, planesA: 1, mathMode: 'argmax', u: 0 }); s.cam = cam3p(); },
        at(s, t) { s.u = t; LA.cam.tween(s, { yaw: -1.25 }, { yaw: -0.4 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Regions from above', duration: 4,
        narration: 'Look straight down and color each point by the winning plane: three <b>decision region</b>s, wedges with straight edges that meet at one point, where all three scores tie. Every linear classifier draws straight edges — always.',
        setup(s) { Object.assign(s, { view: '3d', planesA: 1, regionsA: 0, edges: 0, showBars: false, mathMode: 'edge', u: 0 }); s._c0 = { yaw: s.cam.yaw, pitch: s.cam.pitch }; },
        at(s, t) {
          s.u = t;
          if (t < 0.4) { s.view = '3d'; LA.cam.tween(s, s._c0, { yaw: -Math.PI / 2, pitch: 1.5 }, seg(t, 0, 0.38)); s.planesA = 1 - seg(t, 0.2, 0.4); }
          else { s.view = '2d'; s.cam = cam2(); s.planesA = 0; }
          s.regionsA = seg(t, 0.4, 0.8); s.edges = seg(t, 0.7, 0.95);
        } },

      { title: 'An edge is a p1-09 boundary for the difference', duration: 3.5,
        narration: 'Where do A and B tie? Where (w_A − w_B)·x + (b_A − b_B) = 0: the <b>decision boundary</b> is the p1-09 boundary of the single arrow w_A − w_B. Three arrows give three difference arrows and three edges — multiclass is binary, three times.',
        setup(s) { Object.assign(s, { view: '2d', cam: cam2(), regionsA: 1, edges: 1, diffU: 0, mathMode: 'diff', u: 0 }); },
        at(s, t) { s.u = t; s.diffU = t; } },

      { title: 'Scores to probabilities: the triangle', duration: 4.5,
        narration: 'Softmax (p4-04) turns the probe’s three scores into three probabilities summing to one — a point on the <b>probability triangle</b>. Near an edge the point sits between two corners; at the triple point it is the center (⅓, ⅓, ⅓).',
        setup(s) { Object.assign(s, { diffU: 0, showSimplex: true, showBars: true, mathMode: 'soft', u: 0 }); s._p0 = [1.4, 0.9]; },
        at(s, t) { s.u = t; const tp = triplePoint(s.W, s.b) || [0, 0]; s.probe = lerpV(s._p0, tp, seg(t, 0.1, 0.9)); } },

      { title: 'Confidence is the length of the arrows', duration: 3.5,
        narration: 'Double every arrow. The regions do not move — the ties are the same — but the probabilities sharpen: the triangle point rushes toward a corner. Direction decides the class; length decides how sure. Temperature is the same dial in reverse.',
        setup(s) { Object.assign(s, { probe: [0.8, 0.2], gain: 1, mathMode: 'gain', u: 0 }); },
        at(s, t) { s.u = t; s.gain = 1 + seg(t, 0.15, 0.85); } },

      { title: '✕ Train longer and the edges will curve', duration: 4, misconception: true,
        narration: '“Train longer and the edges will curve.” Swap in the circles data: no choice of arrows separates an inner ring from an outer ring. The edges stay straight and the regions stay wedges, however long you train. To bend a boundary you need a lift first (p2-12, p6-05).',
        setup(s) { Object.assign(s, { preset: 'circles', gain: 1, showSimplex: false, showBars: false, showProbe: false, regionsA: 0, edges: 0, misU: 0, mathMode: 'mis', u: 0 }); },
        at(s, t) { s.u = t; s.misU = seg(t, 0.05, 0.6); } },

      { title: 'Learning rotates the arrows', duration: 5.5,
        narration: 'Back to the blobs, with misplaced arrows (we train on the raw points with every bias b = 0, to keep the picture simple). The loss is <b>cross-entropy</b>, −log p_true; with <b>one-hot</b> targets y its gradient for the weights is (p − y)xᵀ averaged over points — an outer product (p5-06). Each gradient-descent step (p5-03) pulls each arrow toward its own blob.',
        setup(s) { Object.assign(s, { preset: 'blobs', misU: 0, regionsA: 1, edges: 1, showProbe: false, gdK: 0, mathMode: 'gd', u: 0 }); },
        at(s, t) { s.u = t; s.gdK = 10 * seg(t, 0.05, 0.92, 'linear'); } },

      { title: 'The bias, batched', duration: 4,
        narration: 'A bias b_c lifts plane c without tilting it, moving its edges. With the homogeneous trick (p2-11) the bias is a third weight on a constant feature 1, and the whole batch is one product: S = X̃W̃ᵀ with shapes (n×3)(3×3) (p6-01).',
        setup(s) { Object.assign(s, { gdK: -1, W: LA.clone(W_DEF), b: [0, 0, 0], view: '3d', planesA: 1, regionsA: 1, edges: 1, liftB: 0, mathMode: 'bias', u: 0 }); s.cam = cam3p(); },
        at(s, t) { s.u = t; s.b = [0.6 * seg(t, 0.2, 0.75), 0, 0]; LA.cam.tween(s, { yaw: -1.22 }, { yaw: -0.9 }, t); } },

      { title: 'Why ML cares: the last layer of every classifier', duration: 4, mlTie: true,
        narration: 'Every image or text classifier ends with this picture: 1000 class arrows in a 2048-D feature space, regions that are convex cones, softmax over 1000 scores. The features come from the lift before it (p2-12). The picture stops at 3. The formula does not.',
        setup(s) { Object.assign(s, { scene: 'ml', view: '2d', cam: cam2(), b: [0, 0, 0], mathMode: 'ml', u: 0 }); },
        at(s, t) { s.u = t; } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'A linear classifier is a set of arrows. Scores are shadows; the longest shadow wins; ties are straight edges perpendicular to the difference arrows. Softmax puts the scores on the probability triangle, and the gradient (p − y)xᵀ rotates the arrows toward their classes.',
        cards: ['<b>Arrows tile the plane into wedges.</b> Each region is where one class arrow casts the longest shadow.', '<b>Scores → triangle via softmax; length = confidence.</b> Doubling every arrow keeps the regions, sharpens p.', '<b>∂L/∂W = (p − y)xᵀ rotates the arrows</b> toward their own classes, step by step.'],
        setup(s) { Object.assign(s, { scene: 'plane', view: '2d', cam: cam2(), W: LA.clone(W_DEF), b: [0, 0, 0], gain: 1, gdK: -1, regionsA: 1, edges: 1, arrowsA: 1, showProbe: true, probe: [0.8, 0.2], shadows: 0, showBars: true, showSimplex: true, planesA: 0, mathMode: 'soft', u: 1 }); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of the three class arrows and the white probe. Step gradient descent, change the temperature or the bias of class A, or swap the data.</p>',
      setup(s) { Object.assign(s, { scene: 'plane', view: '2d', cam: cam2(), arrowsA: 1, showProbe: true, regionsA: 1, edges: 1, showBars: true, showSimplex: true, dragArrows: true, dragProbe: true, mathMode: 'play', checkMode: 'play' }); },
      controls: [
        { type: 'slider', bind: 'T', min: 0.2, max: 5, step: 0.05, label: 'Temperature T', color: 'gold' },
        { type: 'slider', bind: 'b[0]', min: -2, max: 2, step: 0.05, label: 'Bias of class A' },
        { type: 'button', label: 'One GD step (η = 0.5)', primary: true, onClick(s) { const G = grad(s.W, s.b, dataOf(s)); s.W = N.subM(s.W, N.scaleM(G, ETA)); } },
        { type: 'select', bind: 'view', label: 'View', options: [{ label: 'Regions (2D)', value: '2d' }, { label: 'Score planes (3D)', value: '3d' }] },
        { type: 'select', bind: 'preset', label: 'Data', options: [{ label: 'blobs', value: 'blobs' }, { label: 'circles', value: 'circles' }] },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Trained', apply(s) { s.W = LA.clone(s.Wsol); s.gain = 1; s.b = [0, 0, 0]; } },
          { label: 'Misplaced', apply(s) { s.W = LA.clone(W_BAD); s.gain = 1; s.b = [0, 0, 0]; } },
          { label: 'Confident (gain 3)', apply(s) { s.gain = 3; } },
          { label: 'Reset', apply(s) { s.W = LA.clone(W_DEF); s.gain = 1; s.b = [0, 0, 0]; s.T = 1; s.preset = 'blobs'; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the three arrows so that every point is in its own class’s region.',
        setup(s) { Object.assign(s, { scene: 'plane', view: '2d', cam: cam2(), preset: 'blobs', W: LA.clone(W_BAD), b: [0, 0, 0], gain: 1, gdK: -1, arrowsA: 1, showProbe: false, regionsA: 1, edges: 1, showBars: false, showSimplex: false, dragArrows: true, dragProbe: false, mathMode: 'play', checkMode: 'acc' }); },
        judge(s) { const a = accuracy(s.W, s.b, s.blob, 1); return { ok: a >= 1, progress: clamp01((a - 0.33) / 0.67) }; },
        hint: 'Point each arrow at its blob.',
        solution(s) { s.W = LA.clone(s.Wsol); } },
      { kind: 'manipulate',
        prompt: 'Drag the probe to where the classifier is maximally unsure: all three probabilities within 0.05 of one third.',
        setup(s) { Object.assign(s, { scene: 'plane', view: '2d', cam: cam2(), preset: 'blobs', W: LA.clone(W_DEF), b: [0.3, -0.2, 0.1], gain: 1, gdK: -1, arrowsA: 1, showProbe: true, probe: [1.5, 1.5], regionsA: 1, edges: 1, showBars: true, showSimplex: true, dragArrows: false, dragProbe: true, mathMode: 'soft', checkMode: 'unsure' }); },
        judge(s) { const p = probs(s.W, s.b, s.probe, 1, s.T); const d = Math.max(...p.map(v => Math.abs(v - 1 / 3))); return { ok: d < LA.TOL.scalar, progress: clamp01(1 - d / 0.6) }; },
        hint: 'Find the point where the three edges meet.',
        solution(s) { s.probe = triplePoint(s.W, s.b); } },
      { kind: 'entry',
        prompt: 'Type the difference vector w_A − w_B that is perpendicular to the A|B edge, for w_A = (1, 1) and w_B = (−1.2, 0.3).',
        bind: 'guessDiff', answer: [2.2, 0.7], tol: LA.TOL.entry, label: 'w_A − w_B', colors: 'w',
        setup(s) { Object.assign(s, { scene: 'plane', view: '2d', cam: cam2(), preset: 'blobs', W: LA.clone(W_DEF), b: [0, 0, 0], gain: 1, gdK: -1, arrowsA: 1, showProbe: false, regionsA: 1, edges: 1, showBars: false, showSimplex: false, guessDiff: [1, 0], mathMode: 'guess', checkMode: 'diff' }); } },
    ],
  });

  // ---------- 2D plane ----------
  function edgeRays(W, b, gain) {
    const tp = triplePoint(W, b); if (!tp) return [];
    const out = [];
    [[0, 1, 2], [0, 2, 1], [1, 2, 0]].forEach(([a, c, o]) => {
      const d = N.sub(W[a], W[c]); let dir = [-d[1], d[0]];
      const wa = N.sub(W[a], W[o]);
      if (N.dot(wa, dir) < 0) dir = N.scale(dir, -1);
      out.push({ a, c, from: tp, dir: N.normalize(dir) });
    });
    return out;
  }
  function draw2d(g, s, ctx) {
    const W = Wcur(s), b = s.b, D = dataOf(s);
    g.grid(null, { range: 4, alpha: 0.35, color: 'ghost', axisColor: 'ghostAxis' });
    if (s.binary > 0) {
      const w = W[0];
      g.heat2d(x => (w[0] * x[0] + w[1] * x[1]) >= 0 ? 0 : 1, g.P.bounds(), { res: 64, range: [0, 2], ramp: catRamp, alpha: 0.16 });
      g.lineThrough([0, 0], [-w[1], w[0]], { len: 12, color: 'w', width: 2 });
      g.points(D.X.filter((x, i) => D.y[i] < 2), { r: 3.5, colors: D.X.map((x, i) => i).filter(i => D.y[i] < 2).map(i => CLS[D.y[i]]) });
      g.vector(w, { color: 'w', width: 3, label: 'w' });
      g.label('w·x + b > 0', [1.9, 2.6], { color: 'data', size: 12, font: 'ui' });
      g.label('w·x + b < 0', [-1.9, -2.6], { color: 'dataB', size: 12, font: 'ui' });
      return;
    }
    if (s.misU > 0) { drawMis(g, s); return; }
    if (s.regionsA > 0) g.heat2d(x => argmax(scoresOf(W, b, x, s.gain)), g.P.bounds(), { res: 80, range: [0, 2], ramp: catRamp, alpha: 0.22 * s.regionsA });
    if (s.edges > 0) edgeRays(W, b, s.gain).forEach(r => g.line(r.from, N.add(r.from, N.scale(r.dir, 7 * s.edges)), { color: 'white', width: 1.6, alpha: 0.8 }));
    g.points(D.X, { r: 3.5, colors: D.X.map((x, i) => CLS[D.y[i]]) });
    if (s.edges > 0.9) { const tp = triplePoint(W, b); if (tp) g.point(tp, { r: 4.5, color: 'white', stroke: 'gold' }); }
    const aU = s.arrowsA > 0 ? s.arrowsA : (s.regionsA > 0 ? 1 : 0);
    if (aU > 0) for (let c = 0; c < 3; c++) {
      const tip = N.scale(W[c], s.gain * aU);
      g.vector(tip, { color: CLS[c], width: 3.2, label: NAMES[c], labelSize: 16 });
      if (s.dragArrows) g.handle('w' + c, tip, (p, st) => { st.W[c] = [p[0] / st.gain, p[1] / st.gain]; }, { color: CLS[c] });
    }
    // difference arrows from the triple point
    if (s.diffU > 0) {
      const tp = triplePoint(W, b) || [0, 0];
      const d = N.sub(W[0], W[1]);
      g.vector(tp, N.add(tp, N.scale(d, seg(s.diffU, 0, 0.4))), { color: 'w', width: 3, label: s.diffU > 0.3 ? 'A−B' : undefined, labelSize: 13 });
      if (s.diffU > 0.4) { const r = edgeRays(W, b).find(e => e.a === 0 && e.c === 1); if (r) g.arc(tp, d, r.dir, { r: 0.35, color: 'w', right: true }); }
      if (s.diffU > 0.6 && s.diffU < 0.95) {
        g.vector(tp, N.add(tp, N.sub(W[0], W[2])), { color: 'w', width: 2, alpha: 0.5, label: 'A−C', labelSize: 12 });
        g.vector(tp, N.add(tp, N.sub(W[1], W[2])), { color: 'w', width: 2, alpha: 0.5, label: 'B−C', labelSize: 12 });
      }
    }
    if (s.checkMode === 'diff') {
      // highlight the A|B edge (the guess should stand perpendicular to it); the true difference is not drawn
      const tp = triplePoint(W, b) || [0, 0], r = edgeRays(W, b).find(e => e.a === 0 && e.c === 1);
      if (r) { g.line(tp, N.add(tp, N.scale(r.dir, 7)), { color: 'gold', width: 2.4, alpha: 0.85 }); g.label('A|B edge', N.add(tp, N.scale(r.dir, 2.4)), { color: 'gold', size: 12, font: 'ui', dx: 30 }); }
      g.vector(tp, N.add(tp, s.guessDiff), { color: 'w', width: 3, label: 'guess', labelSize: 12 });
    }
    // probe and its shadows on the three arrow lines
    if (s.showProbe) {
      const x = s.probe;
      if (s.shadows > 0) for (let c = 0; c < 3; c++) {
        const wh = N.normalize(W[c]), foot = N.scale(wh, N.dot(x, wh) * seg(s.shadows, c / 3, (c + 1) / 3));
        if (seg(s.shadows, c / 3, (c + 1) / 3) <= 0) continue;
        g.lineThrough([0, 0], wh, { len: 4, color: CLS[c], alpha: 0.18, width: 1 });
        g.line(x, foot, { color: 'residual', dashed: true, width: 1.2 });
        g.line([0, 0], foot, { color: 'mag', width: 3, alpha: 0.85 });
      }
      g.point(x, { r: 6, color: 'white', label: 'x', labelColor: 'white' });
      if (s.dragProbe) g.handle('probe', x, (p, st) => { st.probe = [p[0], p[1]]; }, { color: 'white' });
    }
    if (s.gdK >= 0) {
      const acc = accuracy(W, [0, 0, 0], s.blob);
      g.hud('gradient-descent step ' + Math.min(10, Math.floor(s.gdK + 1e-9)) + ' of 10 · η = 0.5 · accuracy ' + LA.fmt.pct(acc), 14, 22, { size: 13, color: 'gold' });
      g.hud('mean gradient over the 120 raw (unstandardized) points; b fixed at 0, so edges pass through the origin', 14, 42, { size: 11.5, color: 'muted' });
    }
    if ((s.mathMode === 'edge' || s.mathMode === 'diff') && s.edges > 0.5 && !b.some(v => v !== 0)) g.hud('b = 0 here, so all three edges meet at the origin', 14, 22, { size: 12, color: 'muted' });
    if (s.gain > 1.01 && s.mathMode === 'gain') g.hud('arrows × ' + LA.fmt(s.gain) + ' · same regions', g.size.w - 14, 22, { size: 13, color: 'gold', anchor: 'right' });
    if (s.checkMode === 'acc' || s.checkMode === 'play') g.hud('accuracy ' + LA.fmt.pct(accuracy(s.W, s.b, dataOf(s), s.gain)), g.size.w - 14, 22, { size: 13, color: 'gold', anchor: 'right' });
  }
  function drawMis(g, s) {
    const D = s.circ, u = s.misU;
    // the best a straight split can do (searched once in heavy(): any direction, any offset)
    const M = heavy().mis, n = M.n, c0 = M.c;
    g.heat2d(x => (n[0] * x[0] + n[1] * x[1] - c0) >= 0 ? 0 : 1, g.P.bounds(), { res: 64, range: [0, 2], ramp: catRamp, alpha: 0.18 * u });
    g.lineThrough([n[0] * c0, n[1] * c0], [-n[1], n[0]], { len: 12, color: 'gold', width: 2.4, alpha: u });
    g.points(D.X, { r: 3.5, colors: D.X.map((x, i) => CLS[D.y[i]]) });
    g.circle([0, 0], 1.6, { stroke: 'muted', dashed: true, width: 1.4 });
    g.label('a lifted model (p2-12) bends it like this', [0, -1.6], { anchor: 'below', dy: 8, size: 11.5, color: 'muted', font: 'ui' });
    let ok = 0; D.X.forEach((x, i) => { const cls = (n[0] * x[0] + n[1] * x[1] - c0) >= 0 ? 0 : 1; if (cls === D.y[i]) ok++; });
    g.hud('best straight split (any angle, any offset): accuracy ' + LA.fmt.pct(ok / D.X.length), 14, 22, { size: 13, color: 'gold' });
    if (u > 0.5) { g.marker.cross([22, 52], { screen: true, size: 20 }); g.hud('“edges curve with training”', 38, 52, { size: 13, color: 'text' }); g.hud('straight edges, however long you train', 14, 78, { size: 12, color: 'bad' }); }
  }

  // ---------- 3D: score planes ----------
  function draw3d(g, s) {
    const W = Wcur(s), b = s.b, D = dataOf(s);
    if (s.sheet > 0) {
      const w = W[0];
      g.grid(null, { range: 3, alpha: 0.35 });
      g.surface(x => 2 * N.sigmoid(w[0] * x[0] + w[1] * x[1]), { x0: -3, x1: 3, y0: -3, y1: 3 }, { res: 20, color: 'w', alpha: 0.3, mode: 'both', wireStep: 2 });
      g.points(D.X.filter((x, i) => D.y[i] < 2).map(p => [p[0], p[1], 0]), { r: 3, colors: D.X.map((x, i) => i).filter(i => D.y[i] < 2).map(i => CLS[D.y[i]]) });
      g.vector([0, 0, 0], [w[0], w[1], 0], { color: 'w', width: 3, label: 'w' });
      g.label('σ(w·x + b), drawn ×2 tall', [0, 0, 2.4], { size: 12, color: 'w', font: 'ui' });
      return;
    }
    if (s.regionsA > 0) g.heat3d(x => argmax(scoresOf(W, b, x, s.gain)), { bounds: { x0: -3, x1: 3, y0: -3, y1: 3 }, res: 32, range: [0, 2], ramp: catRamp, alpha: 0.3 * s.regionsA, z: -0.01 });
    else g.grid(null, { range: 3, alpha: 0.35 });
    g.points(D.X.map(p => [p[0], p[1], 0]), { r: 3, colors: D.X.map((x, i) => CLS[D.y[i]]) });
    if (s.planesA > 0.02) {
      const pls = [0, 1, 2].map(c => ({ point: [0, 0, ZS * b[c]], span: [[1, 0, ZS * s.gain * W[c][0]], [0, 1, ZS * s.gain * W[c][1]]], extent: 2.4, color: CLS[c], alpha: 0.2 * s.planesA }));
      g.planes(pls);
      for (let c = 0; c < 3; c++) { const p = N.scale(W[c], 2.1 / (N.norm(W[c]) || 1)); g.label('plane ' + NAMES[c], [p[0], p[1], ZS * (s.gain * N.dot(W[c], p) + b[c])], { color: CLS[c], size: 13, font: 'ui', dy: -10 }); }
      if (s.showProbe && s.planesA > 0.5) {
        const x = s.probe, sc = scoresOf(W, b, x, s.gain), top = argmax(sc);
        g.line([x[0], x[1], 0], [x[0], x[1], ZS * Math.max(...sc, 0)], { color: 'white', width: 1.6, dashed: true });
        sc.forEach((v, c) => g.point([x[0], x[1], ZS * v], { r: c === top ? 6 : 4, color: c === top ? 'gold' : CLS[c], stroke: c === top ? 'white' : undefined }));
        g.point([x[0], x[1], 0], { r: 5, color: 'white', label: 'x', labelColor: 'white' });
      }
      if (s.mathMode === 'bias') g.hud('plane A raised by b_A = ' + LA.fmt(b[0]) + ' (heights ×0.5)', 14, 22, { size: 13, color: 'data' });
      else g.hud('score planes, heights drawn ×0.5', 14, 22, { size: 12, color: 'muted' });
    }
    for (let c = 0; c < 3; c++) g.vector([0, 0, 0], [W[c][0] * s.gain, W[c][1] * s.gain, 0], { color: CLS[c], width: 2.6, label: NAMES[c], labelSize: 14 });
  }

  // ---------- screen insets: score bars, probability triangle, tiles ----------
  function drawInsets(g, s) {
    const W = Wcur(s), x = s.probe;
    if (s.showBars && s.showProbe && !g.is3d) {
      const sc = scoresOf(W, s.b, x, s.gain), top = argmax(sc);
      const bx = 12, by = 12, bw = 168, bh = 118;
      panelBg(g, bx, by, bw, bh);
      txt(g, 'scores at x', bx + 8, by + 14, { size: 11.5 });
      const mx = Math.max(3, ...sc.map(Math.abs)), zy = by + 70, cw = 40;
      sc.forEach((v, c) => {
        const h = Math.abs(v) / mx * 40, xx = bx + 16 + c * (cw + 10);
        rect(g, xx, v >= 0 ? zy - h : zy, cw, Math.max(1, h), C.alpha(C.role(c === top ? 'gold' : CLS[c]), 0.9));
        txt(g, NAMES[c] + ' ' + LA.fmt(v), xx + cw / 2, by + bh - 10, { anchor: 'center', size: 11, color: CLS[c] });
      });
      rect(g, bx + 10, zy, bw - 20, 1, C.muted);
    }
    if (s.showSimplex && s.showProbe && !g.is3d) {
      const p = probs(W, s.b, x, s.gain, s.T);
      const S = g.size, vw = 200, vh = 190, vx = 8, vy = S.h - vh - 8;
      const cam = Object.assign(LA.cam.lookAlong(Object.assign(LA.cam.defaults3d(), { target: [1 / 3, 1 / 3, 1 / 3], dist: 4, fov: 30 }), [1, 1, 1]), { ortho: true, halfH: 0.95 });
      g.viewport({ x: vx, y: vy, w: vw, h: vh }, { view: '3d', cam, border: true, bg: C.alpha(C.panel, 0.92) }, (gg) => {
        const E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        gg.polygon(E, { fill: 'gold', alpha: 0.07, stroke: 'muted', subdiv: 0 });
        E.forEach((e, c) => gg.label(NAMES[c], e, { color: CLS[c], size: 13, font: 'math', dx: (c === 0 ? -10 : c === 1 ? 10 : 14), dy: c === 2 ? 4 : 8 }));
        gg.point([1 / 3, 1 / 3, 1 / 3], { r: 2.5, color: 'muted' });
        gg.point(p, { r: 6, color: 'gold', stroke: 'white' });
      });
      txt(g, 'probability triangle', vx + 8, vy + 12, { size: 11.5, color: 'muted' });
      txt(g, 'p = (' + p.map(v => LA.fmt(v)).join(', ') + ')', vx + 8, vy + vh - 12, { size: 11.5, color: 'gold' });
    }
    if (s.gdK >= 0 && !g.is3d) {
      const k = Math.min(9, Math.floor(s.gdK + 1e-9)), G = s.grads[k];
      const S = g.size, bx = S.w - 138, by = S.h - 120;
      panelBg(g, bx - 18, by - 26, 148, 112);
      txt(g, '∂L/∂W used by step ' + (k + 1), bx - 10, by - 12, { size: 11.5, color: 'hud' });
      G.forEach((row, i) => row.forEach((v, j) => {
        rect(g, bx + j * 52, by + i * 24, 50, 22, C.diverge(clamp01(Math.abs(v) / 0.6) * Math.sign(v)));
        txt(g, LA.fmt(v), bx + j * 52 + 25, by + i * 24 + 11, { anchor: 'center', size: 10.5, color: CLS[i] });
      }));
    }
    if (s.mathMode === 'bias' && g.is3d && s.u > 0.6) {
      const S = g.size, ty = S.h - 70;
      const t1 = { w: 3 * 6, h: 60 };
      rect(g, 20, ty - 30, t1.w, t1.h, C.alpha(C.data, 0.35)); txt(g, '120×3', 20 + t1.w / 2, ty + 42, { anchor: 'center', size: 11, font: 'mono' });
      rect(g, 50, ty - 9, 18, 18, C.alpha(C.w, 0.35)); txt(g, '3×3', 59, ty + 24, { anchor: 'center', size: 11, font: 'mono' });
      txt(g, '→', 78, ty, { size: 15 });
      rect(g, 96, ty - 30, 18, 60, C.alpha(C.gold, 0.35)); txt(g, '120×3', 105, ty + 42, { anchor: 'center', size: 11, font: 'mono' });
      txt(g, 'X̃ W̃ᵀ = S,  with X̃ = [X 1]', 130, ty, { size: 12, color: 'hud' });
    }
  }

  // ---------- mlTie panel ----------
  function drawML(g, s) {
    const S = g.size, Wb = s.Wbig, f = s.feat;
    const cw = Math.min(36, (S.w - 200) / 16), ch = Math.min(28, S.h * 0.38 / 10), x0 = 40, y0 = 70;
    txt(g, 'W: 10 class arrows × 16 features (stand-in for 1000 × 2048, synthetic)', 20, 30, { size: 12.5 });
    Wb.forEach((row, i) => row.forEach((v, j) => rect(g, x0 + j * cw, y0 + i * ch, cw - 1, ch - 1, C.diverge(Math.max(-1, Math.min(1, v / 2))))));
    const hi = 3;
    rect(g, x0 - 2, y0 + hi * ch - 2, 16 * cw + 2, 2, C.gold); rect(g, x0 - 2, y0 + (hi + 1) * ch - 1, 16 * cw + 2, 2, C.gold);
    txt(g, '← one class arrow', x0 + 16 * cw + 8, y0 + (hi + 0.5) * ch, { size: 11.5, color: 'gold' });
    const sc = Wb.map(r => N.dot(r, f) / 2), p = N.softmax(sc, 1), top = argmax(p);
    const by = y0 + 10 * ch + 60, bw = Math.min(64, (S.w - 80) / 10), bh = Math.max(80, Math.min(180, S.h - by - 90));
    txt(g, 'softmax over the 10 scores Wφ(x)', 20, by - 16, { size: 12.5 });
    p.forEach((v, c) => { const h = v * bh / Math.max(...p); rect(g, x0 + c * bw + 3, by + bh - h, bw - 6, Math.max(1, h), C.alpha(c === top ? C.gold : C.data, 0.85)); txt(g, LA.fmt(v), x0 + (c + 0.5) * bw, by + bh - h - 8, { anchor: 'center', size: 9.5 }); txt(g, String(c), x0 + (c + 0.5) * bw, by + bh + 10, { anchor: 'center', size: 10, color: 'muted' }); });
    txt(g, 'The picture stops at 3. The formula does not.', 20, S.h - 44, { size: 13, color: 'gold' });
    txt(g, 'see p6-08: training this classifier', S.w - 20, S.h - 20, { size: 12, color: 'gold', anchor: 'right' });
  }
})();
