/* p4-03 — Diagonalization, matrix powers, power iteration. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, C = LA.color;
  const I2 = () => [[1, 0], [0, 1]];
  const A0 = [[2, 1], [1, 2]];                    // λ = 3 on p₁ = [1; 1], λ = 1 on p₂ = [1; −1]
  const P0 = [[1, 1], [1, -1]], PINV = [[0.5, 0.5], [0.5, -0.5]], D0 = [[3, 0], [0, 1]];
  const P1 = [1, 1], P2 = [1, -1];
  const SHEAR = [[1, 1], [0, 1]];
  const GROW = [[1.3, 0.2], [0.2, 1.3]], SHRINK = [[0.5, 0.2], [0.2, 0.5]], MIXED = [[0.95, 0.45], [0.45, 0.95]];
  const ASLOW = [[2, 0.3], [0.3, 1.6]];           // close second eigenvalue: λ₂/λ₁ ≈ 0.67
  const V0 = [1, -0.6];
  const FAN = Array.from({ length: 24 }, (_, k) => { const a = (k + 0.5) * Math.PI / 12; return [Math.cos(a), Math.sin(a)]; });   // offset 7.5° so no arrow sits exactly on an eigenline
  const pS = k => '<span class="c-gold">p' + k + '</span>', vS = '<span class="c-v">v</span>';

  /** symmetric positive-definite power with real k (pure in k): Q·diag(λᵏ)·Qᵀ */
  function spdPow(A, k) {
    const e = N.eigSym(A), Q = N.fromCols(e.vectors);
    return N.matMul(N.matMul(Q, N.diag(e.values.map(l => Math.pow(l, k)))), N.transpose(Q));
  }
  const lineAngleDeg = (a, d) => { const x = N.angle(a, d) * 180 / Math.PI; return Math.min(x, 180 - x); };
  /** power-iteration record: unit iterates (from LA.num.powerIter history), stretch ratios |Av|, angles to the dominant line */
  function histFor(A, v0, n) {
    const r = N.powerIter(A, n, v0);
    const vs = [N.normalize(v0)].concat(r.history);
    const dom = N.eigSym(A).vectors[0];
    return { vs: vs.map(v => [+v[0].toFixed(5), +v[1].toFixed(5)]), ratios: vs.slice(0, n).map(v => +N.norm(N.matVec(A, v)).toFixed(4)), angles: vs.map(v => +lineAngleDeg(v, dom).toFixed(3)), value: r.value };
  }
  function errRun(A, v0, n) {           // stray/wanted component ratio |tan θ| to the dominant eigenline, k = 0..n
    const e = N.eigSym(A), q1 = e.vectors[0], q2 = e.vectors[1];
    const out = []; let v = N.normalize(v0);
    for (let k = 0; k <= n; k++) { out.push(Math.abs(N.dot(v, q2) / N.dot(v, q1))); v = N.normalize(N.matVec(A, v)); }
    return { errs: out.map(x => +x.toFixed(5)), ratio: e.values[1] / e.values[0] };
  }
  function makeML() {
    // PageRank on 5 pages (column-stochastic link matrix with damping 0.85)
    const L = [[0, 0, 1, 0.5, 0], [0.5, 0, 0, 0, 0], [0.5, 0.5, 0, 0.5, 1], [0, 0.5, 0, 0, 0], [0, 0, 0, 0, 0]];
    const G = L.map(r => r.map(x => 0.85 * x + 0.15 / 5));
    let r = [0.2, 0.2, 0.2, 0.2, 0.2]; const ranks = [r];
    for (let k = 0; k < 10; k++) { r = N.matVec(G, r); ranks.push(r.map(x => +x.toFixed(4))); }
    const W = [[0.8, 0.1, 0.2], [0.1, 0.7, 0.3], [0.1, 0.2, 0.5]];
    let p = [1, 0, 0]; const trail = [p];
    for (let k = 0; k < 12; k++) { p = N.matVec(W, p); trail.push(p.map(x => +x.toFixed(4))); }
    return { ranks, trail, pi: N.stationary(W).map(x => +x.toFixed(4)) };
  }

  LA.registerLesson({
    id: 'p4-03-diagonalization-powers',
    part: 4,
    title: 'Diagonalization, matrix powers, power iteration',
    subtitle: 'A = PDP⁻¹: read in the eigenbasis, stretch, translate back — and powers become powers of numbers.',
    minutes: 10,
    prereqs: ['p4-02-eigenvectors', 'p4-01-change-of-basis'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'PageRank runs power iteration on a 10⁹×10⁹ stochastic matrix; a 100-step RNN applies Wᵏ with k = 100', transfers: 'Aᵏ = PDᵏP⁻¹ and “the largest |λ| wins” hold in any dimension; only the number of eigenlines grows' },
    cantShow: ['non-diagonalizable matrices beyond the 2D shear (we show the one-eigenline case and stop)', 'complex dominant eigenvalues (the fan spirals instead of converging) — mentioned, not animated beyond p4-02'],
    terms: [
      { term: 'diagonalization', def: 'Writing A = PDP⁻¹, with the eigenvectors as the columns of P and the eigenvalues on the diagonal of D.' },
      { term: 'matrix power', def: 'Aᵏ: apply the matrix A k times in a row.' },
      { term: 'dominant eigenvalue', def: 'The eigenvalue of largest size; it controls Aᵏ when k is large.' },
      { term: 'power iteration', def: 'Repeat “multiply by A, divide by the length” until the vector settles on the dominant eigenvector.' },
    ],
    atlas: ['markov', 'backprop', 'pca'],
    thumbnail: 3,

    state() {
      const ml = makeML(), a = errRun(A0, V0, 10), b = errRun(ASLOW, V0, 10);
      return {
        view: '2d', cam: { center: [0.6, 0.6], halfH: 4.2 },
        scene: 'plane', panel: 'pdp',
        A: LA.clone(A0), morph: 0, showStd: true, showGold: true, goldVecs: true,
        v: [2, 0], stage: -1, chainT: -1, k: -1, kMode: 'spd', normalize: true, showBar: false,
        fanAlpha: 0, iterK: -1, v0: LA.clone(V0), hist: histFor(A0, V0, 8), showCharts: false, rateK: -1,
        errA: a.errs, errB: b.errs, ratioA: a.ratio, ratioB: b.ratio,
        threeK: 0, shearMark: false, kHandle: false, Dentry: [[3, 0], [0, 1]], useDentry: false,
        eigView: false, mlT: 0, ranks: ml.ranks, trail: ml.trail, pi: ml.pi,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'three') { renderThree(g, s); return; }
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      renderPlane(g, s, ctx);
    },

    math(s, tex) {
      const P = [], gold = { color: 'gold' };
      const num = (x, o) => '<span class="tx-cell ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';   // plain cells (tex.num re-syncs into matrix cells)
      const inv = x => '<span>' + x + '<sup>&#8202;−1</sup></span>';
      const sup = (x, k) => '<span>' + x + '<sup>' + k + '</sup></span>';
      const Ps = () => tex.sym('P', gold), Ds = () => tex.sym('D', gold);
      switch (s.panel) {
        case 'pdp':
          P.push(tex.eq(tex.sym('A'), tex.op('='), Ps(), Ds(), inv(Ps())));
          P.push(tex.eq(Ps(), tex.op('='), tex.mat(P0, { colors: 'gold' }), Ds(), tex.op('='), tex.mat(D0, { colors: 'gold' }), inv(Ps()), tex.op('='), tex.mat(PINV, { colors: 'gold' })));
          break;
        case 'hops': {
          const c = N.matVec(PINV, s.v), dc = N.matVec(D0, c), Av = N.matVec(P0, dc), st = s.stage;
          P.push(tex.eq(tex.sym('A'), tex.sym('v', { color: 'v' }), tex.op('='), Ps(), tex.paren(Ds() + tex.paren(inv(Ps()) + tex.sym('v', { color: 'v' })))));
          P.push(tex.eq(inv(Ps()), tex.vec(s.v, { color: 'v' }), tex.op('='), st >= 0.5 ? tex.vec(c, gold) : num('?'), tex.op('→'), Ds(), tex.op('·'), tex.op('='), st >= 1.5 ? tex.vec(dc, gold) : num('?'), tex.op('→'), Ps(), tex.op('·'), tex.op('='), st >= 2.5 ? tex.vec(Av, { color: 'v' }) : num('?')));
          break; }
        case 'chain': {
          const kk = chainK(s.chainT);
          P.push(tex.eq(sup(tex.sym('A'), 'k'), tex.op('='), Ps(), sup(Ds(), 'k'), inv(Ps())));
          P.push(tex.eq(sup(Ds(), kk), tex.op('='), tex.mat([[Math.pow(3, kk), 0], [0, 1]], { colors: 'gold' }), tex.text('= diag(3' + supStr(kk) + ', 1' + supStr(kk) + ')')));
          break; }
        case 'fan': {
          const k = Math.max(0, s.k);
          P.push(tex.eq(sup(tex.sym('A'), 'k'), tex.sym('v', { color: 'v' }), tex.op('='), tex.sym('c', { sub: '1' }), sup(num(3), 'k'), tex.sym('p', { sub: '1', color: 'gold' }), tex.op('+'), tex.sym('c', { sub: '2' }), sup(num(1), 'k'), tex.sym('p', { sub: '2', color: 'gold' }), tex.op('≈'), tex.sym('c', { sub: '1' }), sup(num(3), 'k'), tex.sym('p', { sub: '1', color: 'gold' })));
          P.push(tex.eq(tex.text('k ='), num(LA.fmt.fixed(k, 1)), tex.text('3ᵏ ='), num(Math.round(Math.pow(3, k)), gold)));
          break; }
        case 'three':
          P.push(tex.eq(tex.abs(tex.sym('λ')), tex.op('>'), num(1), tex.text('explodes'), tex.op('·'), tex.abs(tex.sym('λ')), tex.op('<'), num(1), tex.text('vanishes'), tex.op('·'), tex.sym('λ'), tex.op('='), num(1), tex.text('freezes')));
          P.push(tex.eq(tex.text('shrink:'), tex.mat(SHRINK), tex.sym('λ'), tex.op('='), num(0.7), tex.op(','), num(0.3)));
          break;
        case 'shear':
          P.push(tex.eq(tex.sym('S'), tex.op('='), tex.mat(SHEAR, { colors: 'cols' }), tex.sym('λ'), tex.op('='), num(1), tex.op(','), num(1)));
          P.push(tex.eq(tex.text('only one eigenvector'), tex.vec([1, 0], gold), tex.text('→ no P, no D', { color: 'bad' })));
          P.push(tex.eq(sup(tex.sym('S'), 'k'), tex.op('='), tex.mat([[1, Math.max(0, s.k)], [0, 1]], { digits: 1 })));
          break;
        case 'iter': {
          const h = s.hist, j = Math.min(h.ratios.length - 1, Math.max(0, Math.floor(s.iterK)));
          P.push(tex.eq(tex.sym('v', { color: 'v' }), tex.op('←'), tex.frac(tex.sym('A') + tex.sym('v', { color: 'v' }), tex.norm(tex.sym('A') + tex.sym('v', { color: 'v' })))));
          P.push(tex.eq(tex.text('step ' + (j + 1) + ':'), tex.frac(tex.norm(tex.sym('A') + tex.sym('v', { color: 'v' })), tex.norm(tex.sym('v', { color: 'v' }))), tex.op('='), num(h.ratios[j], gold), tex.op('→'), tex.sym('λ', { sub: '1' }), tex.op('='), num(3, gold)));
          P.push(tex.eq(tex.text('angle to p₁:'), num(LA.fmt.fixed(h.angles[Math.min(h.angles.length - 1, j + 1)], 1) + '°', gold)));
          break; }
        case 'rate':
          P.push(tex.eq(tex.text('error after k steps'), tex.op('∝'), sup(tex.paren(tex.frac(tex.sym('λ', { sub: '2' }), tex.sym('λ', { sub: '1' }))), 'k')));
          P.push(tex.eq(tex.text('A:'), num(LA.fmt.fixed(s.ratioA, 2), gold), tex.text('A′ = [[2, 0.3], [0.3, 1.6]]:'), num(LA.fmt.fixed(s.ratioB, 2), { color: 'b' })));
          break;
        case 'ml':
          P.push(tex.eq(sup(tex.sym('P'), 'k'), tex.sym('π', { sub: '0' }), tex.op('→'), tex.sym('π', gold)));
          P.push(tex.eq(tex.frac(tex.text('∂L', { html: false }), tex.text('∂h₀')), tex.op('∝'), sup(tex.paren('<span>' + tex.T(tex.sym('W')) + '</span>'), 'k')));
          break;
        case 'check2': {
          const A = N.matMul(N.matMul(P0, s.Dentry), PINV);
          P.push(tex.eq(tex.sym('A'), tex.op('='), Ps(), Ds(), inv(Ps()), tex.op('='), tex.mat(A, { colors: 'cols' })));
          break; }
        case 'check3': {
          P.push(tex.eq(tex.text('k ='), num(Math.round(s.k)), tex.text('largest angle to the p₁ line:'), num(LA.fmt.fixed(fanSpread(s.A, Math.round(s.k)), 1) + '°', gold)));
          break; }
        case 'check1': {
          const g10 = N.norm(N.matVec(N.matPow(s.A, 10), s.v)) / Math.max(1e-9, N.norm(s.v));
          P.push(tex.eq(tex.norm(sup(tex.sym('A'), '10') + tex.sym('v', { color: 'v' })), tex.op('/'), tex.norm(tex.sym('v', { color: 'v' })), tex.op('='), num(g10 > 999 ? Math.round(g10) : LA.fmt.fixed(g10, 2), { color: g10 <= 1.05 ? 'ok' : 'bad' })));
          break; }
        case 'play': {
          const e = N.eig2(s.A), k = Math.round(Math.max(0, s.k));
          P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols' }), sup(tex.sym('A'), k), tex.op('='), tex.mat(N.matPow(s.A, k), { digits: 2 })));
          if (e.real && !e.defective) P.push(tex.eq(sup(Ds(), k), tex.op('='), tex.mat([[Math.pow(e.values[0], k), 0], [0, Math.pow(e.values[1], k)]], { colors: 'gold', digits: 2 }), tex.text('λ = ' + LA.fmt(e.values[0]) + ', ' + LA.fmt(e.values[1]))));
          else P.push(tex.text(e.real ? 'one eigenline only: not diagonalizable' : 'complex λ: no real eigenbasis', { color: 'bad' }));
          const h = N.powerIter(s.A, Math.max(1, k), s.v0);
          P.push(tex.eq(tex.text('power-iteration estimate after ' + Math.max(1, k) + ' steps:'), num(h.value, gold)));
          break; }
        default:
          P.push(tex.eq(sup(tex.sym('A'), 'k'), tex.op('='), Ps(), sup(Ds(), 'k'), inv(Ps())));
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Recall: in the eigenbasis A is just two numbers', duration: 3.5,
        narration: 'A = [[2, 1], [1, 2]] stretches ' + pS('₁') + ' = [1; 1] by 3 and ' + pS('₂') + ' = [1; −1] by 1. In the gold basis the whole map is “×3 along p₁, ×1 along p₂”: the diagonal matrix D = [[3, 0], [0, 1]]. From p4-01, A = PDP⁻¹ with P = [p₁ p₂]. Watch the gold grid only stretch while the blue grid shears.',
        setup(s) { Object.assign(s, { scene: 'plane', view: '2d', panel: 'pdp', A: LA.clone(A0), morph: 0, showStd: true, showGold: true, goldVecs: true, stage: -1, chainT: -1, k: -1, fanAlpha: 0, iterK: -1, showCharts: false, rateK: -1, shearMark: false, showBar: false }); s.cam = { center: [0.6, 0.6], halfH: 4.2 }; },
        at(s, t) { s.morph = seg(t, 0.15, 0.9); } },

      { title: 'Three moves, read right to left', duration: 5,
        narration: 'A = PDP⁻¹ is three moves. P⁻¹ reads ' + vS + ' in gold coordinates; D stretches each gold coordinate by its λ; P translates back. Watch ' + vS + ' = [2; 0] take the three hops — [1; 1], then [3; 1], then back — and land on Av = [4; 2], the same place one application of A puts it.',
        setup(s) { s.panel = 'hops'; s.morph = 0; s.v = [2, 0]; s.stage = 0; s.showStd = true; s.goldVecs = false; s.cam = { center: [1.7, 0.9], halfH: 3.3 }; },
        at(s, t) { s.stage = 3 * seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Powers are free: Aᵏ = PDᵏP⁻¹', duration: 4.5,
        narration: 'Apply A ten times? In gold coordinates that is “×3 ten times, ×1 ten times”: powers of two numbers, D¹⁰ = [[59049, 0], [0, 1]]. Write the ten copies of PDP⁻¹ side by side and every inner P⁻¹P cancels. <b>Aᵏ = PDᵏP⁻¹</b>: a <b>matrix power</b> without ten matrix multiplications.',
        setup(s) { s.panel = 'chain'; s.stage = -1; s.chainT = 0; s.morph = 0; s.showStd = false; s.showGold = true; s.goldVecs = false; },
        at(s, t) { s.chainT = t; } },

      { title: 'Repeated application: the fan collapses', duration: 5,
        narration: 'Apply A again and again to a fan of 24 unit arrows (redrawn at length 1 so they fit). The component along ' + pS('₁') + ' grows by 3 each time; the component along p₂ stays. After a few steps every arrow is almost parallel to p₁. The largest eigenvalue wins: the <b>dominant eigenvalue</b>.',
        setup(s) { Object.assign(s, { panel: 'fan', chainT: -1, showGold: false, showStd: false, goldVecs: false, morph: 0, k: 0, kMode: 'spd', normalize: true, fanAlpha: 1, showBar: true }); s.cam = { center: [0, 0], halfH: 2.4 }; },
        at(s, t) { const kk = 6 * seg(t, 0.04, 0.96, 'linear'), f = Math.floor(kk); s.k = Math.min(6, f + seg(kk - f, 0, 0.6, 'inOut')); } },

      { title: 'Explode, vanish, or settle: the size of λ', duration: 4.5,
        narration: 'Lengths depend on |λ|. Top: both eigenvalues above 1 — everything explodes out of the frame. Middle: A = [[0.5, 0.2], [0.2, 0.5]] with λ = 0.7 and 0.3 — everything shrinks to the origin. Bottom: one above, one below — the fan flattens onto a line and grows along it. Exactly λ = 1 would freeze that direction.',
        setup(s) { s.scene = 'three'; s.panel = 'three'; s.threeK = 0; s.showBar = false; },
        at(s, t) { s.threeK = 6 * seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Misconception: every matrix can be diagonalized', duration: 4, misconception: true,
        narration: '✕ The shear S = [[1, 1], [0, 1]] has eigenvalue 1 twice but only <i>one</i> eigenline, the gold ' + '<span class="c-i">î</span>' + ' axis. P needs two eigenvector columns and only one exists: no P, no D. Powers still make sense — the arrows lean toward ' + '<span class="c-i">î</span>' + ' but never reach it — yet the shortcut fails. Symmetric matrices never have this problem.',
        setup(s) { Object.assign(s, { scene: 'plane', panel: 'shear', A: LA.clone(SHEAR), k: 0, kMode: 'shear', normalize: true, fanAlpha: 1, showGold: false, showStd: true, shearMark: false, showBar: false }); s.cam = { center: [0, 0], halfH: 2.4 }; },
        at(s, t) { s.k = 6 * seg(t, 0.05, 0.8); s.shearMark = t > 0.55; } },

      { title: 'Power iteration: normalize every step', duration: 6, interactive: true,
        narration: 'Turn the collapse into a method. Start anywhere, multiply by A, divide by the length so it never explodes, repeat. The yellow arrow settles on ' + pS('₁') + ', and the length before dividing settles on λ₁ = 3. That is <b>power iteration</b>: the top eigenvector without any formula. Drag the start vector and watch it again.',
        setup(s) { Object.assign(s, { panel: 'iter', A: LA.clone(A0), k: -1, fanAlpha: 0, showStd: false, showGold: false, shearMark: false, iterK: 0, v0: LA.clone(V0), hist: histFor(A0, V0, 8), showCharts: true }); s.cam = { center: [0.6, -0.2], halfH: 3.3 }; },
        at(s, t) { s.iterK = 7 * seg(t, 0.03, 0.97, 'linear'); } },

      { title: 'How fast? The ratio λ₂/λ₁', duration: 4,
        narration: 'Each step shrinks the unwanted component by |λ₂/λ₁| relative to the wanted one: here 1/3, so the error drops to a third per step — a straight gold line on this log scale, matching the dashed (1/3)ᵏ. A close second eigenvalue converges slowly: the orange A′ has λ₂/λ₁ ≈ 0.67. That ratio decides how fast a Markov chain forgets its start.',
        setup(s) { s.panel = 'rate'; s.showCharts = false; s.iterK = 7; s.rateK = 0; },
        at(s, t) { s.rateK = 10 * seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Why ML cares: PageRank, Markov chains, exploding gradients', duration: 4, mlTie: true,
        narration: 'PageRank is power iteration on the web’s link matrix, whose dominant eigenvalue is 1. A Markov chain forgets its start at rate λ₂. And an RNN applies the same W at every time step: Wᵏ explodes or vanishes gradients exactly like the fans you just saw. Lesson p4-05 builds the first two.',
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.rateK = -1; s.iterK = -1; s.mlT = 0; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'A = PDP⁻¹: read in the eigenbasis, stretch, translate back. Aᵏ = PDᵏP⁻¹ — powers of numbers. The largest |λ| dominates: explode above 1, vanish below. Power iteration — multiply, normalize, repeat — finds the top eigenvector.',
        cards: ['<b>PDP⁻¹.</b> P⁻¹ reads gold coordinates, D stretches them, P translates back.', '<b>Largest |λ| wins.</b> Aᵏ = PDᵏP⁻¹: every fan collapses onto the dominant eigenline.', '<b>Multiply, normalize, repeat.</b> Power iteration lands on the top eigenvector; |Av| → λ₁.'],
        setup(s) { s.scene = 'recap'; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Edit A, slide k, drag the yellow start vector. Gold lines are the eigenlines; the yellow ghosts are power-iteration steps. Turn on the eigenbasis view to see the same picture in gold coordinates, where A is diagonal.</p>',
      setup(s) { Object.assign(s, { scene: 'plane', view: '2d', panel: 'play', A: LA.clone(A0), k: 3, kMode: 'int', normalize: true, fanAlpha: 0.8, showGold: false, showStd: true, goldVecs: false, morph: 0, iterK: -1, v0: LA.clone(V0), showCharts: false, playIter: true }); s.cam = { center: [0, 0], halfH: 2.8 }; },
      controls: [
        { type: 'matrix', bind: 'A', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'slider', bind: 'k', min: 0, max: 12, step: 1, label: 'k (apply k times)', color: 'gold', fmt: x => String(Math.round(x)) },
        { type: 'toggle', bind: 'normalize', label: 'Normalize each step (draw at length 1)' },
        { type: 'toggle', bind: 'eigView', label: 'Eigenbasis view (gold coordinates)' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Symmetric [[2,1],[1,2]]', values: { A: A0 } },
          { label: 'Grow', values: { A: GROW } },
          { label: 'Shrink', values: { A: SHRINK } },
          { label: 'Mixed', values: { A: MIXED } },
          { label: 'Shear', values: { A: SHEAR } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the yellow start vector so that applying A = [[2, 1], [1, 2]] ten times does not grow it.',
        setup(s) { checkBase(s); s.v = [1, 0.3]; s.panel = 'check1'; s.check1 = true; },
        judge(s) {
          const n = N.norm(s.v); if (n < 0.5) return { ok: false, progress: 0, hint: 'Keep the vector longer than 0.5.' };
          const d = lineAngleDeg(s.v, P2);
          return { ok: d < LA.TOL.angleDeg, progress: Math.max(0, 1 - d / 45) };
        },
        hint: 'Only one direction is left alone by A: the gold line with λ = 1, along [1; −1]. Any other direction picks up a 3¹⁰ = 59049 component.',
        solution(s) { s.v = [1, -1]; } },
      { kind: 'entry',
        prompt: 'Type D so that A = PDP⁻¹ <i>halves</i> everything along p₁ = [1; 1] and <i>doubles</i> everything along p₂ = [1; −1]. The fan under A updates live.',
        bind: 'Dentry', answer: [[0.5, 0], [0, 2]], tol: LA.TOL.entry, label: 'D', colors: ['gold', 'gold'],
        setup(s) { checkBase(s); s.useDentry = true; s.Dentry = [[1, 0], [0, 1]]; s.fanAlpha = 0.9; s.k = 1; s.kMode = 'int'; s.normalize = false; s.showGold = true; s.panel = 'check2'; } },
      { kind: 'manipulate',
        prompt: 'Drag the gold k dot on the track below the fan to the <i>smallest</i> k that brings every arrow of Aᵏ·fan within 5° of the gold ' + pS('₁') + ' line (A = [[2, 1], [1, 2]]). The readout above the track shows the largest angle.',
        setup(s) { checkBase(s); s.k = 0; s.kMode = 'int'; s.normalize = true; s.fanAlpha = 1; s.kHandle = true; s.panel = 'check3'; s.cam = { center: [0, -0.4], halfH: 2.6 }; },
        judge(s) {
          const k = Math.round(s.k), sp = fanSpread(s.A, k), prev = k > 0 ? fanSpread(s.A, k - 1) : 90;
          if (sp < 5 && prev < 5) return { ok: false, progress: 0.8, hint: 'Already below 5° — does a smaller k also work?' };
          return { ok: sp < 5, progress: Math.max(0, 1 - (sp - 5) / 85) };
        },
        hint: 'Each step divides the stray component by 3. The worst arrow starts almost on p₂, so it needs a few more steps than the others.',
        solution(s) { s.k = 5; } },
    ],
  });

  function checkBase(s) {
    Object.assign(s, { scene: 'plane', view: '2d', A: LA.clone(A0), morph: 0, showStd: true, showGold: false, goldVecs: false, stage: -1, chainT: -1, k: -1, fanAlpha: 0, iterK: -1, showCharts: false, rateK: -1, shearMark: false, showBar: false, kHandle: false, useDentry: false, check1: false, playIter: false, eigView: false });
    s.cam = { center: [0, 0], halfH: 2.8 };
  }
  const SUPC = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const supStr = k => String(k).split('').map(c => SUPC[c] || c).join('');
  const chainK = t => Math.max(1, Math.min(10, 1 + Math.round(9 * seg(t, 0.62, 0.95, 'linear'))));
  function fanSpread(A, k) { let m = 0; const Ak = N.matPow(A, k); FAN.forEach(u => { m = Math.max(m, lineAngleDeg(N.matVec(Ak, u), P1)); }); return m; }
  function powM(s) {
    const A = s.useDentry ? N.matMul(N.matMul(P0, s.Dentry), PINV) : s.A;
    const k = Math.max(0, s.k);
    if (s.kMode === 'spd') return spdPow(A, k);
    if (s.kMode === 'shear') return [[1, k], [0, 1]];
    return N.matPow(A, Math.round(k));
  }

  /* ---------------- the main 2D plane ---------------- */
  function renderPlane(g, s, ctx) {
    const A = s.useDentry ? N.matMul(N.matMul(P0, s.Dentry), PINV) : s.A;
    const e = N.eig2(A);
    // eigenbasis view (playground): everything is drawn through V = P⁻¹ of the current eigenvectors
    let V = I2();
    if (s.eigView && e.real && !e.defective) { const Pm = N.fromCols(e.vectors), Pi = N.inv(Pm); if (Pi) V = Pi; }
    const T = p => N.matVec(V, p);
    const F = lerpM(I2(), A, s.morph);
    g.ghostGrid();
    if (s.showStd) g.grid(N.matMul(V, F), { alpha: s.k >= 0 || s.iterK >= 0 ? 0.35 : 0.8 });
    if (s.showGold) g.grid(N.matMul(V, N.matMul(F, P0)), { color: 'gold', alpha: 0.5, width: 1.2 });
    if (s.eigView) g.hud('eigenbasis view: gold coordinates — A acts as D, a pure stretch', 14, 22, { color: 'gold', size: 13 });

    // eigenlines (gold)
    const showLines = s.k >= 0 || s.iterK >= 0 || s.stage >= 0 || s.check1;
    const lines = s.useDentry ? [N.normalize(P1), N.normalize(P2)] : (e.real ? e.vectors : []);
    if (showLines) lines.forEach((q, i) => g.lineThrough([0, 0], T(q), { color: 'gold', width: i === 0 ? 2.2 : 1.4, alpha: i === 0 ? 0.9 : 0.55 }));
    if (s.useDentry || s.check1) {
      g.label('p₁', T(s.check1 ? [1.05, 1.05] : [1.7, 1.7]), { color: 'gold', size: 13.5, font: 'ui', dx: 24 });
      g.label('p₂', T([-1.7, 1.7]), { color: 'gold', size: 13.5, font: 'ui', dx: -26 });
    } else if (showLines && (s.kMode === 'spd' || s.kHandle) && s.k >= 0 && lines.length === 2) {
      g.label('p₁ (×3)', T(N.scale(lines[0], 1.75)), { color: 'gold', size: 13, font: 'ui', dx: 30 });
      g.label('p₂ (×1)', T(N.scale(lines[1], 1.75)), { color: 'gold', size: 13, font: 'ui', dx: 30 });
    }
    if (s.goldVecs) {
      g.vector(T(N.matVec(F, P1)), { color: 'gold', width: 3, label: 'p₁', labelFont: 'math' });
      g.vector(T(N.matVec(F, P2)), { color: 'gold', width: 3, label: 'p₂', labelFont: 'math' });
      if (s.morph > 0.95) { g.label('×3', T(N.matVec(F, P1)), { color: 'gold', size: 13, font: 'ui', bold: true, dx: -24, dy: -12 }); g.label('×1', T(N.matVec(F, P2)), { color: 'gold', size: 13, font: 'ui', bold: true, dx: 22, dy: 4 }); }
    }

    // three hops (step 2) and the cancelling chain (step 3)
    if (s.stage >= 0) drawHops(g, s);
    if (s.chainT >= 0) drawChain(g, s);

    // matrix power fan
    if (s.fanAlpha > 0.01 && s.k >= 0) {
      const M = powM(s);
      let maxLen = 0;
      FAN.forEach(u => {
        const w = N.matVec(M, u), L = N.norm(w); maxLen = Math.max(maxLen, L);
        const tip = s.normalize ? (L > 1e-12 ? N.scale(w, 1 / L) : w) : w;
        g.vector(T(u), { color: 'residual', dashed: true, width: 1, head: 5, alpha: 0.3 });
        g.vector(T(tip), { color: 'muted', width: 1.5, head: 7, alpha: 0.9 * s.fanAlpha });
      });
      if (s.normalize) g.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1, strokeAlpha: 0.6 });
      if (s.showBar) {
        const W = g.size.w, H = g.size.h;
        g.bars([Math.max(1, maxLen)], { screen: true, x: W - 70, y: 60, w: 26, h: H - 200, max: Math.pow(3, 6), color: 'gold' });
        g.hud('true length', W - 96, 40, { color: 'muted', size: 11.5 });
        g.hud(LA.fmt(Math.round(maxLen)), W - 92, H - 122, { color: 'gold', size: 13, bold: true });
        g.hud('k = ' + LA.fmt.fixed(Math.max(0, s.k), 1), 14, 22, { color: 'hud', size: 13 });
        g.hud('arrows drawn at length 1; the bar shows the real length (3ᵏ)', 14, 42, { color: 'muted', size: 12 });
      }
      if (s.kMode === 'shear') {
        g.label('only eigenline', [-1.95, 0], { color: 'gold', size: 12.5, font: 'ui', dy: -14, anchor: 'left' });
        drawShearTiles(g, s);
      }
      if (s.useDentry) g.ellipse(N.matMul(V, A), [0, 0], { stroke: 'teal', width: 1.8 });
      if (s.kHandle) drawKTrack(g, s);
    }

    // power iteration (steps 7–8)
    if (s.iterK >= 0) drawIter(g, s, T);
    if (s.showCharts) drawIterCharts(g, s);
    if (s.rateK >= 0) drawRate(g, s);

    // playground: power iteration from the yellow start vector + its handle
    if (s.playIter || s.check1) {
      const v = s.check1 ? s.v : s.v0;
      if (s.playIter && N.norm(v) > 1e-6) {
        const n = Math.round(Math.max(0, s.k)), h = N.powerIter(A, Math.max(1, n), v).history;
        let prev = N.normalize(v);
        for (let j = 0; j < Math.min(n, h.length); j++) { g.vector(T(h[j]), { color: 'v', width: 1.4, head: 7, alpha: 0.25 + 0.5 * (j + 1) / n }); prev = h[j]; }
        if (n > 0) g.label('v after ' + n, T(prev), { color: 'v', size: 12, font: 'ui', dy: -14 });
      }
      if (s.check1) {
        const A10 = N.matVec(N.matPow(A, 10), v), L = N.norm(A10);
        if (L > 1e-9) g.vector(T(N.scale(A10, Math.min(2.5, L) / L)), { color: 'v', dashed: true, width: 2, alpha: 0.6, label: 'A¹⁰v (direction)', labelFont: 'ui', labelSize: 12 });
      }
      g.vector(T(v), { color: 'v', width: 3, label: 'v', labelFont: 'math' });
      const Vi = N.inv(V) || I2();
      g.handle('v0', T(v), (p, st) => { const q = N.matVec(Vi, p); if (st.check1) st.v = q; else st.v0 = q; }, { color: 'v' });
    }
  }

  function drawHops(g, s) {
    const st = s.stage, v = s.v, c = N.matVec(PINV, v);       // c = [1, 1]
    const u2 = seg(st, 1, 2, 'inOut'), u3 = seg(st, 2, 3, 'inOut');
    const c1 = c[0] * (1 + 2 * u2), c2 = c[1];                    // D stretches the gold coordinates: 1 → 3 along p₁, 1 → 1 along p₂
    const Av = N.matVec(A0, v);
    // direct hop for comparison
    g.vector(v, Av, { color: 'hud', dashed: true, width: 1.4, head: 8, alpha: 0.6 });
    g.vector(v, { color: 'v', width: 3.2, label: 'v', labelFont: 'math' });
    const a1 = seg(st, 0, 0.6);
    if (a1 > 0.01) {
      const P1v = N.scale(P1, c1), tip = N.add(P1v, N.scale(P2, c2));
      g.vector([0, 0], P1v, { color: 'gold', width: 2.8, head: 11, alpha: a1, label: LA.fmt(c1) + '·p₁', labelAt: 0.55, labelOffset: -16, labelFont: 'ui', labelSize: 12.5 });
      g.vector(P1v, tip, { color: 'gold', width: 2.8, head: 11, alpha: a1, label: LA.fmt(c2) + '·p₂', labelAt: 0.5, labelOffset: 16, labelFont: 'ui', labelSize: 12.5 });
      if (u3 > 0.01) g.vector(N.scale(tip, u3), { color: 'v', width: 3.2, alpha: u3, label: u3 > 0.9 ? 'Av' : '', labelFont: 'math' });
      if (st > 2.95) g.marker.ring(Av, { r: 14 });
    }
    // inset: the same hops in gold coordinates (where the gold grid is square)
    const W = g.size.w;
    g.viewport({ x: W - 200, y: g.size.h - 200, w: 188, h: 188 }, { view: '2d', cam: { center: [1.6, 0.6], halfH: 2.1 }, border: true, bg: 'panel' }, (g2, r) => {
      g2.grid(null, { color: 'gold', alpha: 0.35, width: 1, range: 5 });
      const cc = [c1, c2];
      if (a1 > 0.01) g2.vector(cc, { color: 'gold', width: 2.6, head: 10, alpha: a1, label: '[' + LA.fmt(c1) + '; ' + LA.fmt(c2) + ']', labelFont: 'ui', labelSize: 12 });
      g2.hud('gold coordinates', r.x + 8, r.y + 14, { size: 11.5, color: 'gold' });
      g2.hud(st < 1 ? 'hop 1: P⁻¹ reads v' : st < 2 ? 'hop 2: D: ×3 and ×1' : 'hop 3: P maps back', r.x + 8, r.y + r.h - 12, { size: 11, color: 'hud' });
    });
  }

  function drawChain(g, s) {
    const t = s.chainT, W = g.size.w, H = g.size.h;
    const toks = ['P', 'D', 'P⁻¹', 'P', 'D', 'P⁻¹', 'P', 'D', 'P⁻¹'], inner = { 2: 1, 3: 1, 5: 1, 6: 1 };
    const hl = t > 0.06 && t < 0.42, fade = 1 - seg(t, 0.3, 0.52);
    const slot = i => (toks[i].length > 1 ? 50 : 36) * (inner[i] ? fade : 1);
    let total = 0; toks.forEach((_, i) => { total += slot(i); });
    let x = W / 2 - total / 2 - 20; const y = H * 0.3;
    toks.forEach((tk, i) => {
      const w = slot(i), a = inner[i] ? fade : 1;
      if (a > 0.02) g.label(tk, [x + w / 2, y], { screen: true, size: 30, font: 'math', bold: true, color: inner[i] && hl ? 'b' : 'gold', alpha: a });
      x += w;
    });
    g.label('· · ·', [x + 22, y], { screen: true, size: 26, color: 'muted', font: 'ui' });
    if (hl) [2, 5].forEach((i, n) => g.label('P⁻¹P = I', [W / 2 - 60 + n * 110, y + 40], { screen: true, size: 13, color: 'b', font: 'ui' }));
    if (t > 0.52) {
      const kk = chainK(t);
      g.label('A' + supStr(kk) + ' = P D' + supStr(kk) + ' P⁻¹', [W / 2, H * 0.47], { screen: true, size: 30, color: 'gold', font: 'math', bold: true, alpha: seg(t, 0.52, 0.62) });
      g.label('D' + supStr(kk) + ' = diag(3' + supStr(kk) + ', 1' + supStr(kk) + ') = diag(' + Math.pow(3, kk) + ', 1)', [W / 2, H * 0.58], { screen: true, size: 18, color: 'hud', font: 'ui', alpha: seg(t, 0.55, 0.65) });
      g.label('only two numbers get powered', [W / 2, H * 0.65], { screen: true, size: 13, color: 'muted', font: 'ui', alpha: seg(t, 0.6, 0.7) });
    }
  }

  function drawShearTiles(g, s) {
    if (!s.shearMark) return;
    const x = 18, y = g.size.h - 92;
    g.hud('P = [', x, y, { color: 'gold', size: 22, font: 'math' });
    g.hud('î', x + 64, y, { color: 'i', size: 22, font: 'math' });
    g.hud('|', x + 88, y, { color: 'muted', size: 22 });
    g.hud('?', x + 104, y, { color: 'bad', size: 22, font: 'math', bold: true });
    g.hud(']', x + 126, y, { color: 'gold', size: 22, font: 'math' });
    g.marker.cross([x + 110, y], { screen: true, size: 30 });
    g.hud('no second eigenvector to put here', x, y + 30, { color: 'bad', size: 12.5 });
  }

  function drawKTrack(g, s) {
    if (g.is3d) return;
    const W = g.size.w, H = g.size.h, x0 = 40, x1 = W - 60, y = H - 44, kmax = 8;
    g.hud('k', x0 - 22, y, { color: 'gold', size: 14, font: 'math' });
    for (let k = 0; k <= kmax; k++) g.hud(String(k), x0 + (x1 - x0) * k / kmax - 4, y + 18, { color: 'muted', size: 11 });
    const P = g.P, kk = Math.round(LA.clamp(s.k, 0, kmax));
    const a = P.toWorld(x0, y), b = P.toWorld(x1, y);
    g.line(a, b, { color: 'hud', width: 2 });
    const pos = P.toWorld(x0 + (x1 - x0) * kk / kmax, y);
    g.point(pos, { r: 6, color: 'gold' });
    g.handle('k', pos, (p, st) => { const sx = P.toScreen(p)[0]; st.k = Math.round(LA.clamp((sx - x0) / (x1 - x0) * kmax, 0, kmax)); }, { color: 'gold' });
    g.hud('spread around p₁: ' + LA.fmt.fixed(fanSpread(s.A, kk), 1) + '°', x0 - 22, y - 26, { color: fanSpread(s.A, kk) < 5 ? 'ok' : 'hud', size: 13 });
  }

  function drawIter(g, s, T) {
    const h = s.hist, n = h.ratios.length;
    const kk = Math.min(n - 0.001, Math.max(0, s.iterK)), j = Math.floor(kk), f = kk - j;
    g.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1, strokeAlpha: 0.6 });
    for (let i = 0; i < j; i++) g.vector(T(h.vs[i]), { color: 'v', width: 1.4, head: 7, alpha: 0.18 + 0.3 * i / Math.max(1, j) });
    const vj = h.vs[j], Avj = N.matVec(s.A, vj);
    if (f < 0.5) {
      const w = lerpV(vj, Avj, seg(f, 0.05, 0.45, 'inOut'));
      g.vector(T(w), { color: 'v', dashed: true, width: 2.2, alpha: 0.8, label: f > 0.2 ? 'Av' : '', labelFont: 'math' });
      g.vector(T(vj), { color: 'v', width: 3.2, label: 'v', labelFont: 'math' });
    } else {
      const w = lerpV(Avj, h.vs[j + 1], seg(f, 0.55, 0.95, 'inOut'));
      g.vector(T(vj), { color: 'v', width: 1.6, alpha: 0.35 });
      g.vector(T(w), { color: 'v', width: 3.2, label: f > 0.8 ? 'v' : 'Av/|Av|', labelFont: f > 0.8 ? 'math' : 'ui' });
    }
    g.vector(T(P1), { color: 'gold', width: 2.6, head: 10, label: 'p₁', labelFont: 'math', dy: -12 });
    // draggable start vector (interactive step / playground)
    g.vector(T(s.v0), { color: 'b', width: 2, head: 9, alpha: 0.7, label: 'start', labelFont: 'ui', labelSize: 12 });
    g.handle('start', T(s.v0), (p, st) => { if (N.norm(p) < 0.15) return; st.v0 = p; st.hist = histFor(st.A, p, 8); }, { color: 'b' });
  }

  function drawIterCharts(g, s) {
    const h = s.hist, W = g.size.w, H = g.size.h;
    const upto = Math.min(h.ratios.length, Math.floor(Math.max(0, s.iterK)) + 1);
    const w = Math.min(230, (W - 70) / 2), ch = 86, y = H - ch - 30;
    const ang = h.angles.slice(0, upto + 1).map((a, i) => [i, a]);
    const rat = h.ratios.slice(0, upto).map((r, i) => [i + 1, r]);
    g.lineChart([{ points: ang, color: 'gold', width: 2 }], { x: 20, y, w, h: ch, xRange: [0, 8], yRange: [0, 90], title: 'angle to p₁ (degrees)', xLabel: 'step' });
    g.lineChart([{ points: rat.length > 1 ? rat : [[1, rat.length ? rat[0][1] : 0], [1.01, rat.length ? rat[0][1] : 0]], color: 'v', width: 2 }], { x: 40 + w, y, w, h: ch, xRange: [1, 8], yRange: [0, 3.4], title: '|Av| / |v|  →  λ₁ = 3', xLabel: 'step', hline: 3, hlineColor: 'gold' });
  }

  function drawRate(g, s) {
    const W = g.size.w, H = g.size.h, k = s.rateK, n = Math.floor(k);
    const x = 30, w = W - 60, ch = 190, y = H - ch - 40;
    // log scale (values transformed before g.lineChart): geometric decay becomes a straight line whose slope is λ₂/λ₁
    const lg = x => Math.log10(Math.max(1e-9, x));
    const pts = (arr) => arr.slice(0, n + 1).map((e, i) => [i, lg(e)]).concat(n < 10 ? [[k, LA.lerp(lg(arr[n]), lg(arr[Math.min(10, n + 1)]), k - n)]] : []);
    const ref = (e0, r) => [[0, lg(e0) + 0.12], [Math.max(0.01, k), lg(e0 * Math.pow(r, k)) + 0.12]];
    g.lineChart([
      { points: ref(s.errA[0], s.ratioA), color: 'gold', dash: [5, 4], width: 1.4 },
      { points: ref(s.errB[0], s.ratioB), color: 'b', dash: [5, 4], width: 1.4 },
      { points: pts(s.errA), color: 'gold', width: 2.4, label: 'A: ×' + LA.fmt.fixed(s.ratioA, 2) + ' per step' },
      { points: pts(s.errB), color: 'b', width: 2.4, label: 'A′: ×' + LA.fmt.fixed(s.ratioB, 2) + ' per step' },
    ], { x, y, w: w - 130, h: ch, xRange: [0, 10], yRange: [-4.6, 1], title: 'stray ÷ wanted component, log scale (each gridline = ÷10); dashed: (λ₂/λ₁)ᵏ', xLabel: 'step k' });
    for (let d = 0; d >= -4; d--) {
      const yy = y + ch - (d + 4.6) / 5.6 * ch;
      g.hud(d === 0 ? '1' : '10' + supStr(-d).replace(/^/, '⁻'), x - 26, yy, { color: 'muted', size: 10.5 });
      g.hud('·', x + 2, yy, { color: 'muted', size: 10 });
    }
  }

  /* ---------------- three fans: grow / shrink / mixed ---------------- */
  function renderThree(g, s) {
    const rows = [{ M: GROW, name: 'λ = 1.5, 1.1: explodes' }, { M: SHRINK, name: 'λ = 0.7, 0.3: vanishes' }, { M: MIXED, name: 'λ = 1.4, 0.5: flattens onto a line, grows along it' }];
    rows.forEach((row, i) => {
      const Mk = spdPow(row.M, s.threeK);
      let maxL = 0; FAN.forEach(u => { maxL = Math.max(maxL, N.norm(N.matVec(Mk, u))); });
      const out = i === 0 && maxL > 2.3;
      g.viewport({ x: 0, y: i / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 2.4 }, border: true, bg: out ? 'rgba(255,92,92,0.10)' : undefined }, (g2, r) => {
        g2.ghostGrid();
        const e = N.eigSym(row.M);
        e.vectors.forEach(q => g2.lineThrough([0, 0], q, { color: 'gold', width: 1.3, alpha: 0.6 }));
        g2.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1 });
        FAN.forEach((u, idx) => { if (idx % 2) return; g2.vector(N.matVec(Mk, u), { color: 'muted', width: 1.5, head: 7 }); });
        g2.hud(row.name, r.x + 12, r.y + 18, { size: 13, color: out ? 'bad' : 'hud', bold: true });
        g2.hud('k = ' + LA.fmt.fixed(s.threeK, 1) + '   longest arrow: ' + LA.fmt(maxL), r.x + 12, r.y + 38, { size: 12, color: 'muted' });
      });
    });
  }

  /* ---------------- ML tie-in thumbnails ---------------- */
  function renderML(g, s) {
    const u = s.mlT;
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [2.5, 0.9], halfH: 1.3 }, border: true }, (g2, r) => {
      const kk = 10 * u, k0 = Math.floor(kk), f = kk - k0;
      const ranks = s.ranks[k0].map((x, i) => LA.lerp(x, s.ranks[Math.min(10, k0 + 1)][i], f));
      g2.bars(ranks, { origin: [0, 0], width: 0.7, gap: 0.3, height: 1.6, max: 0.45, colors: ['data', 'data', 'gold', 'data', 'data'], labels: ['A', 'B', 'C', 'D', 'E'], values: true });
      g2.hud('PageRank: power iteration on 5 pages, step ' + k0, r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
      g2.hud('dominant eigenvalue 1 → the bars settle', r.x + 12, r.y + 36, { size: 11.5, color: 'muted' });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0.15], halfH: 1.25 }, border: true }, (g2, r) => {
      const c = [[-1.2, -0.8], [1.2, -0.8], [0, 1.0]], to2 = p => [p[0] * c[0][0] + p[1] * c[1][0] + p[2] * c[2][0], p[0] * c[0][1] + p[1] * c[1][1] + p[2] * c[2][1]];
      g2.polygon(c, { fill: 'teal', alpha: 0.18, stroke: 'teal' });
      ['sunny', 'cloudy', 'rainy'].forEach((nm, i) => g2.label(nm, c[i], { color: 'muted', size: 11.5, font: 'ui', dy: i === 2 ? 4 : 12, dx: i === 2 ? 30 : 0 }));
      const kk = 12 * u, k0 = Math.floor(kk), f = kk - k0;
      const pts = s.trail.slice(0, k0 + 1).map(to2);
      const cur = to2(lerpV(s.trail[k0], s.trail[Math.min(12, k0 + 1)], f));
      g2.polyline(pts.concat([cur]), { color: 'hud', width: 1.4, alpha: 0.8 });
      pts.forEach(p => g2.point(p, { r: 2.5, color: 'hud' }));
      g2.point(to2(s.pi), { r: 7, color: 'gold' });
      g2.point(cur, { r: 5, color: 'white' });
      g2.hud('Markov chain: the state slides to the gold fixed point', r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 1.5 }, border: true }, (g2, r) => {
      const Q = N.rot2(0.5), k = 8 * u;
      const Wk = N.matMul(N.matMul(Q, N.diag([Math.pow(0.92, k), Math.pow(0.55, k)])), N.transpose(Q));
      g2.lineThrough([0, 0], N.col(Q, 0), { color: 'gold', width: 1.3, alpha: 0.7 });
      g2.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1 });
      g2.ellipse(Wk, [0, 0], { stroke: 'w', width: 2.2, fill: 'w', alpha: 0.12 });
      g2.hud('RNN: Wᵏ with |λ| < 1 squeezes the circle (vanishing gradient)   k = ' + LA.fmt.fixed(k, 1), r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
    });
  }

  /* ---------------- recap cards ---------------- */
  function renderRecap(g, s) {
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 2 }, border: true }, (g2, r) => {
      const y = r.y + r.h * 0.52, x = r.x + r.w / 2, dx = Math.min(130, r.w * 0.22);
      [['P', 'translate back'], ['D', 'stretch ×λ'], ['P⁻¹', 'read gold coords']].forEach((t, i) => {
        const cx = x + (i - 1) * dx;
        g2.label(t[0], [cx, y - 6], { screen: true, size: 32, color: 'gold', font: 'math', bold: true });
        g2.label(t[1], [cx, y + 28], { screen: true, size: 12, color: 'muted', font: 'ui' });
      });
      g2.hud('A = PDP⁻¹ (read right to left)', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 1.4 }, border: true }, (g2, r) => {
      g2.lineThrough([0, 0], P1, { color: 'gold', width: 2, alpha: 0.9 });
      const M = spdPow(A0, 4);
      FAN.forEach(u => g2.vector(N.normalize(N.matVec(M, u)), { color: 'muted', width: 1.3, head: 6 }));
      g2.label('p₁', [0.9, 0.9], { color: 'gold', size: 14, font: 'math', dx: 14 });
      g2.hud('Largest |λ| wins: A⁴ · fan', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0.3, 0], halfH: 1.3 }, border: true }, (g2, r) => {
      g2.lineThrough([0, 0], P1, { color: 'gold', width: 2, alpha: 0.9 });
      g2.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1 });
      const h = histFor(A0, V0, 6);
      h.vs.forEach((v, i) => g2.vector(v, { color: 'v', width: i === h.vs.length - 1 ? 3 : 1.4, head: i === h.vs.length - 1 ? 11 : 7, alpha: i === h.vs.length - 1 ? 1 : 0.25 + 0.1 * i, label: i === 0 ? 'start' : (i === h.vs.length - 1 ? 'v' : ''), labelFont: 'ui', labelSize: 12 }));
      g2.hud('Multiply, normalize, repeat', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
  }
})();
