/* p4-04 — Probability vectors, the simplex, softmax, stochastic matrices. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, seg = LA.seg, C = LA.color;
  const E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const CENTER = [1 / 3, 1 / 3, 1 / 3];
  const PW = [[0.8, 0.1, 0.2], [0.1, 0.7, 0.3], [0.1, 0.2, 0.5]];      // weather chain, columns sum to 1
  const Z0 = [2, 1, 0];
  const NAMES = ['cat', 'dog', 'bird'], COLS = ['i', 'j', 'k'];
  const BAD_HI = [0.6, 0.5, 0.1], BAD_NEG = [0.7, -0.1, 0.4];
  const T_LO = 0.1, T_HI = 5;
  const C2 = [[-1, -0.72], [1, -0.72], [0, 1]];                          // the triangle drawn flat in 2D thumbnails
  const to2 = p => [p[0] * C2[0][0] + p[1] * C2[1][0] + p[2] * C2[2][0], p[0] * C2[0][1] + p[1] * C2[1][1] + p[2] * C2[2][1]];
  const vS = '<span class="c-v">p</span>';

  const sum3 = p => p[0] + p[1] + p[2];
  /** clamp a point to the simplex: project onto the plane Σ = 1, clip negatives, renormalize */
  function toSimplex(p) {
    const d = (sum3(p) - 1) / 3; let q = p.map(x => Math.max(0, x - d)); const s = sum3(q);
    return s > 1e-9 ? q.map(x => x / s) : CENTER.slice();
  }
  function colNormalize(P) {
    const out = P.map(r => r.map(x => Math.max(0, x)));
    for (let j = 0; j < 3; j++) { let s = 0; for (let i = 0; i < 3; i++) s += out[i][j]; for (let i = 0; i < 3; i++) out[i][j] = s > 1e-9 ? out[i][j] / s : 1 / 3; }
    return out;
  }
  const fmtP = p => '[' + p.map(x => LA.fmt.fixed(x, 2)).join(', ') + ']';
  const tOf = u => T_LO * Math.pow(T_HI / T_LO, u), uOf = T => Math.log(T / T_LO) / Math.log(T_HI / T_LO);
  function tempPath(z) { const out = []; for (let i = 0; i <= 40; i++) { const T = 0.05 * Math.pow(400, i / 40); out.push(N.softmax(z, T).map(x => +x.toFixed(4))); } return out; }

  LA.registerLesson({
    id: 'p4-04-simplex-softmax',
    part: 4,
    title: 'Probability vectors, the simplex, softmax, stochastic matrices',
    subtitle: 'Where probabilities live, how softmax puts any scores there, and why a stochastic matrix must have a fixed point.',
    minutes: 10,
    prereqs: ['p1-03-linear-combinations-span', 'p2-02-matrix-vector', 'p4-02-eigenvectors'],
    tags: ['core'],
    badge: { drawnIn: '3D (3 classes → a triangle)', realSize: 'a language model’s softmax has 50 000 entries; a 1000-class classifier lives on a 999-dimensional simplex', transfers: '“non-negative, sums to 1” is the whole definition; softmax and temperature act entrywise the same way for any length' },
    cantShow: ['the simplex for 4 classes is a tetrahedron (drawable), for 5+ it is not; we stop at the triangle', 'why softmax specifically (maximum entropy / log-linear reasons): geometry only here'],
    terms: [
      { term: 'probability vector', def: 'A list of numbers that are all ≥ 0 and add up to 1.' },
      { term: 'simplex', def: 'The set of all probability vectors: a triangle for 3 entries, a tetrahedron for 4.' },
      { term: 'one-hot', def: 'A corner of the simplex: all the probability on one class.' },
      { term: 'softmax', def: 'Exponentiate each score and divide by the total, which always lands on the simplex.' },
      { term: 'temperature', def: 'A number T that divides the scores before softmax: small T sharpens toward a corner, large T flattens toward the center.' },
      { term: 'stochastic matrix', def: 'A square matrix whose columns are probability vectors; it maps the simplex into itself.' },
    ],
    atlas: ['softmax', 'markov', 'attention'],
    thumbnail: 5,

    state() {
      return {
        view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [0.36, 0.36, 0.3], dist: 4.4, yaw: 0.785, pitch: 0.49 }),
        scene: 'main', panel: 'p',
        p: [0.5, 0.3, 0.2], showP: 1, drops: 1, bars: 1, barsMode: 'p',
        planeA: 0, triA: 0, labelsA: 0, wrong: 0, okMark: false,
        z: LA.clone(Z0), T: 1, stage: -1, showSoft: false, tempOn: false, tPath: tempPath(Z0), satT: -1, tHandle: false,
        P: LA.clone(PW), imageT: -1, showImage: false, sample: [0.15, 0.25, 0.6], mlT: 0, pHandle: false,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      renderMain(g, s, ctx);
    },

    math(s, tex) {
      const P = [], gold = { color: 'gold' };
      const num = (x, o) => '<span class="tx-cell ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';   // plain cells (tex.num re-syncs into matrix cells)
      const pvec = v => tex.mat(v.map(x => [x]), { colors: (i) => COLS[i] });
      switch (s.panel) {
        case 'p':
          P.push(tex.eq(tex.sym('p'), tex.op('='), pvec(s.p)));
          P.push(tex.eq(tex.sym('p', { sub: 'i' }), tex.op('≥'), num(0), tex.op(','), tex.text('Σ'), tex.sym('p', { sub: 'i' }), tex.op('='), num(LA.fmt.fixed(sum3(s.p), 2), gold)));
          break;
        case 'simplex':
          P.push(tex.eq(tex.sym('Δ'), tex.op('='), tex.text('{ p : pᵢ ≥ 0, Σ pᵢ = 1 }')));
          P.push(tex.eq(tex.text('corners (one-hot)'), pvec([1, 0, 0]), pvec([0, 1, 0]), pvec([0, 0, 1]), tex.text('center'), pvec([1 / 3, 1 / 3, 1 / 3])));
          break;
        case 'wrong':
          P.push(tex.eq(pvec(BAD_HI), tex.text('Σ = 1.2', { color: 'bad' }), tex.text('✕', { color: 'bad' })));
          P.push(tex.eq(pvec(BAD_NEG), tex.text('p₂ < 0', { color: 'bad' }), tex.text('✕', { color: 'bad' })));
          break;
        case 'softmax': {
          const ez = s.z.map(Math.exp), sm = N.softmax(s.z, 1), st = s.stage;
          P.push(tex.eq(tex.text('softmax(z)ᵢ'), tex.op('='), tex.frac(tex.text('e^zᵢ'), tex.text('Σⱼ e^zⱼ'))));
          P.push(tex.eq(tex.sym('z'), tex.op('='), pvec(s.z), tex.op('→'), tex.text('eᶻ'), st >= 0.5 ? pvec(ez) : num('?'), tex.op('→'), tex.text('÷ ' + LA.fmt(N.sum(ez))), st >= 1.5 ? pvec(sm) : num('?')));
          break; }
        case 'temp': {
          const sm = N.softmax(s.z, s.T);
          P.push(tex.eq(tex.text('softmax(z / T)'), tex.op('='), pvec(sm), tex.text('T ='), num(LA.fmt.fixed(s.T, 2), gold)));
          P.push(tex.eq(tex.text('T → 0: one-hot'), tex.op('·'), tex.text('T → ∞: uniform')));
          break; }
        case 'sat': {
          const sm = N.softmax(s.z, 1);
          P.push(tex.eq(tex.text('softmax'), tex.paren(pvec(s.z)), tex.op('='), tex.mat(sm.map(x => [x.toFixed(3)]), { colors: (i) => COLS[i] })));
          P.push(tex.eq(tex.text('attention uses'), tex.text('scores / √d', gold), tex.text('to stay off the corners')));
          break; }
        case 'stoch':
          P.push(tex.eq(tex.sym('P'), tex.op('='), tex.mat(s.P, { colors: 'w' }), tex.text('columns sum to 1')));
          P.push(tex.eq(tex.sym('P'), tex.sym('p'), tex.op('='), tex.sym('p', { sub: '1' }), tex.text('col₁ +'), tex.sym('p', { sub: '2' }), tex.text('col₂ +'), tex.sym('p', { sub: '3' }), tex.text('col₃')));
          P.push(tex.eq(tex.sym('P'), tex.sym('π', gold), tex.op('='), tex.sym('π', gold), tex.op('='), tex.mat(N.stationary(s.P).map(x => [x]), { colors: 'gold' })));
          break;
        case 'ml':
          P.push(tex.eq(tex.text('softmax(QKᵀ / √d)'), tex.op('·'), tex.sym('π', { sub: 't+1' }), tex.op('='), tex.sym('P'), tex.sym('π', { sub: 't' })));
          break;
        case 'check2':
          P.push(tex.eq(tex.text('softmax'), tex.paren(pvec(s.z)), tex.op('='), pvec(N.softmax(s.z, 1)), tex.text('target [0.5, 0.25, 0.25]')));
          break;
        case 'play': {
          const sm = N.softmax(s.z, s.T), pi = N.stationary(s.P);
          P.push(tex.eq(tex.sym('p'), tex.op('='), pvec(s.p), tex.text('softmax(z / T)'), tex.op('='), tex.mat(sm.map(x => [x]), { colors: 'gold' })));
          P.push(tex.eq(tex.sym('P'), tex.op('='), tex.mat(s.P, { colors: 'w' }), tex.sym('π', gold), tex.op('='), tex.mat(pi.map(x => [x]), { colors: 'gold' })));
          break; }
        default:
          P.push(tex.eq(tex.text('Σ'), tex.sym('p', { sub: 'i' }), tex.op('='), num(1)));
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Three numbers that are probabilities', duration: 3,
        narration: 'A classifier says: cat 0.5, dog 0.3, bird 0.2. Three numbers, none negative, adding to 1 — a <b>probability vector</b>. Draw it as a point in 3D: cat along the green <span class="c-i">axis 1</span>, dog along the red <span class="c-j">axis 2</span>, bird along the blue <span class="c-k">axis 3</span>.',
        setup(s) { Object.assign(s, { scene: 'main', view: '3d', panel: 'p', p: [0.5, 0.3, 0.2], showP: 0, drops: 0, bars: 0, barsMode: 'p', planeA: 0, triA: 0, labelsA: 0, wrong: 0, okMark: false, stage: -1, showSoft: false, tempOn: false, satT: -1, imageT: -1, pHandle: false, tHandle: false }); s.cam = Object.assign(LA.cam.defaults3d(), { target: [0.36, 0.36, 0.3], dist: 4.4, yaw: 0.785, pitch: 0.49 }); },
        at(s, t) { s.bars = seg(t, 0, 0.35); s.showP = seg(t, 0.35, 0.6); s.drops = seg(t, 0.55, 0.9); } },

      { title: 'The simplex: a triangle in space', duration: 4,
        narration: '“Sums to 1” is a flat plane: x + y + z = 1. “None negative” cuts it down to the triangle with corners (1, 0, 0), (0, 1, 0), (0, 0, 1). Every probability vector is a point on this teal triangle — the <b>simplex</b>. Its corners are certainty (<b>one-hot</b>); its center [⅓, ⅓, ⅓] is total ignorance.',
        setup(s) { s.panel = 'simplex'; s.planeA = 0; s.triA = 0; s.labelsA = 0; s.drops = 0.4; },
        at(s, t) { s.planeA = seg(t, 0, 0.3) * (1 - 0.75 * seg(t, 0.55, 0.8)); s.triA = seg(t, 0.35, 0.65); s.labelsA = seg(t, 0.6, 0.85); } },

      { title: 'Reading a point', duration: 3, interactive: true,
        narration: 'Drag the white point around the triangle. Near a corner, that class dominates. On an edge, one class has probability exactly 0 — the edge lights up. The three numbers say how close you are to each corner; the bars on the right follow along.',
        setup(s) { s.panel = 'p'; s.planeA = 0.25; s.triA = 1; s.labelsA = 1; s.drops = 0.4; s.pHandle = true; s._p0 = [0.5, 0.3, 0.2]; },
        at(s, t) { const a = 2 * Math.PI * seg(t, 0.1, 0.9, 'inOut'); s.p = toSimplex([s._p0[0] + 0.2 * Math.sin(a), s._p0[1] + 0.15 * Math.sin(2 * a), s._p0[2] - 0.2 * Math.sin(a) - 0.15 * Math.sin(2 * a)]); } },

      { title: 'Misconception: any three positive numbers will do', duration: 4, misconception: true,
        narration: '✕ [0.6, 0.5, 0.1] looks fine but sums to 1.2: it floats above the plane. ✕ [0.7, −0.1, 0.4] sums to 1 but has a negative entry: it lies on the plane but outside the triangle. Both are off the simplex; neither is a probability vector. The white point on the triangle is the real thing.',
        setup(s) { s.panel = 'wrong'; s.pHandle = false; s.p = [0.5, 0.3, 0.2]; s.wrong = 0; s.okMark = false; s.planeA = 0.25; },
        at(s, t) { s.wrong = seg(t, 0.25, 0.45) * (1 - 0.3 * seg(t, 0.85, 1)); s.okMark = t > 0.75; s.planeA = 0.25 + 0.35 * seg(t, 0.3, 0.5); LA.cam.tween(s, { yaw: 0.785, pitch: 0.49 }, { yaw: 0.4, pitch: 0.03 }, seg(t, 0, 0.3)); } },

      { title: 'Softmax: any scores → the triangle', duration: 5,
        narration: 'A model outputs <i>scores</i> z = [2, 1, 0]: any real numbers, not probabilities. <b>Softmax</b> fixes that in two moves: exponentiate each (all positive: 7.39, 2.72, 1), then divide by the total 11.1. Result [0.67, 0.24, 0.09]: on the triangle, order preserved, biggest score biggest share.',
        setup(s) { Object.assign(s, { panel: 'softmax', wrong: 0, okMark: false, planeA: 0.12, z: LA.clone(Z0), T: 1, stage: 0, barsMode: 'soft', showP: 0, drops: 0 }); s.cam = Object.assign(LA.cam.defaults3d(), { target: [0.36, 0.36, 0.3], dist: 4.4, yaw: 0.785, pitch: 0.49 }); },
        at(s, t) { s.stage = 2 * seg(t, 0.05, 0.85, 'linear'); s.showSoft = t > 0.85; } },

      { title: 'Temperature', duration: 5, interactive: true,
        narration: 'Divide the scores by a <b>temperature</b> T before softmax. Small T exaggerates differences: the gold point races to the cat corner. Large T washes them out: the point slides to the center. T is a dial between “argmax” and “uniform”. Drag the T dot along the bottom track yourself.',
        setup(s) { Object.assign(s, { panel: 'temp', stage: -1, barsMode: 'temp', showSoft: true, tempOn: true, tHandle: true, T: T_LO, showP: 0 }); },
        at(s, t) { s.T = tOf(seg(t, 0.05, 0.6, 'inOut')) * (1 - seg(t, 0.7, 0.95)) + seg(t, 0.7, 0.95) * 1; } },

      { title: 'Big scores saturate', duration: 3.5,
        narration: 'Multiply the scores by 5: z = [10, 5, 0]. Softmax gives [0.993, 0.007, 0.000]: the point is pinned in a corner and barely moves when the scores wobble. Dot products between long vectors are big, so attention divides scores by √d to keep softmax off the corners — that is p6-03’s √d.',
        setup(s) { Object.assign(s, { panel: 'sat', tempOn: false, tHandle: false, T: 1, satT: 0, barsMode: 'temp' }); },
        at(s, t) { s.satT = t; const k = 1 + 4 * seg(t, 0, 0.35), w = seg(t, 0.4, 1, 'linear'); const env = 0.5 * Math.sin(Math.PI * w);   // the wobble dies out, so the step ends exactly on z = [10, 5, 0]
          s.z = [Z0[0] * k + env * Math.sin(6 * Math.PI * w), Z0[1] * k + env * Math.cos(6 * Math.PI * w + 1), 0]; } },

      { title: 'A stochastic matrix maps the triangle into itself', duration: 5.5,
        narration: 'A <b>stochastic matrix</b> has probability vectors for columns: each column is a purple point on the triangle. Pp = p₁·col₁ + p₂·col₂ + p₃·col₃ is a weighted mix of those three points, so it lands inside the smaller purple triangle they form. The whole triangle is mapped into itself and shrunk; something must stay fixed — an eigenvector with λ = 1.',
        setup(s) { Object.assign(s, { panel: 'stoch', satT: -1, z: LA.clone(Z0), showSoft: false, barsMode: 'none', P: LA.clone(PW), imageT: 0, showP: 0, planeA: 0 }); },
        at(s, t) { s.imageT = t; } },

      { title: 'Why ML cares: everything that is “a probability” lives here', duration: 4, mlTie: true,
        narration: 'Classifier outputs, attention weights for one query, a Markov chain’s state, a language model’s next-token distribution: all are points on a simplex, all produced or moved by softmax and stochastic matrices. The Markov capstone (p4-05) walks the point to its fixed point; attention (p6-03) builds the softmax row from dot products.',
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.imageT = -1; s.mlT = 0; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Probability vectors live on the simplex: a triangle for three classes. Softmax exponentiates and normalizes any scores onto it; temperature slides between a corner and the center. A stochastic matrix maps the triangle into itself, so a fixed point with λ = 1 exists.',
        cards: ['<b>Triangle of probabilities.</b> Entries ≥ 0 that sum to 1: a point on the simplex; corners are one-hot.', '<b>exp, then divide.</b> softmax(z/T) puts any scores on the triangle; T slides between corner and center.', '<b>Triangle into itself ⇒ λ = 1.</b> A stochastic matrix shrinks the simplex into itself; a fixed point exists.'],
        setup(s) { s.scene = 'recap'; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the white point on the triangle. Edit the scores z and slide T: the gold point is softmax(z/T). Edit the stochastic matrix P (columns are renormalized to sum to 1) and show its image triangle and fixed point. Drag to orbit.</p>',
      setup(s) { Object.assign(s, { scene: 'main', view: '3d', panel: 'play', p: [0.5, 0.3, 0.2], showP: 1, drops: 0.4, bars: 1, barsMode: 'p', planeA: 0.12, triA: 1, labelsA: 1, z: LA.clone(Z0), T: 1, showSoft: true, tempOn: true, pHandle: true, tHandle: false, P: LA.clone(PW), showImage: false, imageT: -1 }); },
      controls: [
        { type: 'matrix', bind: 'z', label: 'scores z', colors: ['i', 'j', 'k'], step: 0.1, min: -5, max: 5 },
        { type: 'slider', bind: 'T', min: 0.1, max: 5, step: 0.01, label: 'temperature T', color: 'gold', fmt: x => LA.fmt.fixed(x, 2) },
        { type: 'matrix', bind: 'P', label: 'stochastic P', colors: 'w', step: 0.05, min: 0, max: 1 },
        { type: 'toggle', bind: 'showImage', label: 'Show P’s image triangle P(Δ)' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Scores [2,1,0]', values: { z: Z0, T: 1 } },
          { label: 'Saturated [10,5,0]', values: { z: [10, 5, 0], T: 1 } },
          { label: 'Uniform P', apply(s) { s.P = [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]; s.showImage = true; } },
          { label: 'Weather P', apply(s) { s.P = LA.clone(PW); s.showImage = true; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the white point to where cat and dog are equally likely and bird has probability 0.5.',
        setup(s) { checkBase(s); s.p = [0.5, 0.3, 0.2]; s.showP = 1; s.pHandle = true; s.bars = 1; s.barsMode = 'p'; s.panel = 'p'; },
        judge(s) { return LA.judge.near(s.p, [0.25, 0.25, 0.5], LA.TOL.pos); },
        hint: 'Halfway between the cat and dog corners, then halfway from there to the bird corner.',
        solution(s) { s.p = [0.25, 0.25, 0.5]; } },
      { kind: 'entry',
        prompt: 'Type scores z = [z₁, 0, 0] (keep the last two at 0) whose softmax is [0.5, 0.25, 0.25]. Cat needs twice dog’s share, so e^z₁ must be twice e⁰ = 1. The gold point moves as you type.',
        bind: 'z', answer: [0.69, 0, 0], tol: LA.TOL.entry, label: 'z', colors: ['i', 'j', 'k'],
        setup(s) { checkBase(s); s.z = [0, 0, 0]; s.T = 1; s.showSoft = true; s.bars = 1; s.barsMode = 'temp'; s.panel = 'check2'; s.target = [0.5, 0.25, 0.25]; } },
      { kind: 'manipulate',
        prompt: 'Raise the temperature (drag the T dot) until no class has probability above 0.5, with z = [2, 1, 0].',
        setup(s) { checkBase(s); s.z = LA.clone(Z0); s.T = 1; s.showSoft = true; s.tempOn = true; s.tHandle = true; s.bars = 1; s.barsMode = 'temp'; s.panel = 'temp'; },
        judge(s) { const m = Math.max.apply(null, N.softmax(s.z, s.T)); return { ok: m < 0.5, progress: Math.max(0, Math.min(1, (0.9 - m) / 0.4)) }; },
        hint: 'Flatten the bars: a bigger T pulls the gold point toward the center.',
        solution(s) { s.T = 2.5; } },
    ],
  });

  function checkBase(s) {
    Object.assign(s, { scene: 'main', view: '3d', showP: 0, drops: 0.4, bars: 0, barsMode: 'p', planeA: 0.1, triA: 1, labelsA: 1, wrong: 0, okMark: false, stage: -1, showSoft: false, tempOn: false, satT: -1, imageT: -1, showImage: false, pHandle: false, tHandle: false, target: null });
    s.cam = Object.assign(LA.cam.defaults3d(), { target: [0.36, 0.36, 0.3], dist: 4.4, yaw: 0.785, pitch: 0.49 });
  }

  /* ---------------- the 3D scene ---------------- */
  function renderMain(g, s, ctx) {
    // auto-renormalize the stochastic matrix's columns (playground edits)
    if (ctx.mode === 'play') { const Pn = colNormalize(s.P); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (Math.abs(Pn[i][j] - s.P[i][j]) > 1e-9) s.P[i][j] = Pn[i][j]; }
    g.grid(null, { range: 1.5, step: 0.5, alpha: 0.35 });
    // positive axes, colored by class
    for (let i = 0; i < 3; i++) {
      g.line([0, 0, 0], N.scale(E[i], -0.4), { color: 'gridAxis', width: 1, dashed: true, alpha: 0.5 });
      g.vector([0, 0, 0], N.scale(E[i], 1.35), { color: COLS[i], width: 2, head: 9, label: NAMES[i], labelFont: 'ui', labelSize: 13 });
    }
    // the plane x + y + z = 1 and the triangle
    if (s.planeA > 0.01) g.planes([{ point: CENTER, normal: [1, 1, 1], extent: 1.0, color: 'teal', alpha: 0.12 * s.planeA, stroke: false }]);
    if (s.triA > 0.01) {
      g.polygon(E, { fill: 'teal', alpha: 0.25 * s.triA, stroke: 'teal', width: 1.6 });
      if (s.labelsA > 0.01) {
        E.forEach((c, i) => g.label(NAMES[i] + ' (one-hot)', c, { color: COLS[i], size: 12, font: 'ui', dy: i === 2 ? -16 : 16, alpha: s.labelsA }));
        g.point(CENTER, { r: 3.5, color: 'muted', alpha: s.labelsA });
        if (s.imageT < 0 && !s.showImage && s.wrong < 0.01) g.label('uniform', CENTER, { color: 'muted', size: 11.5, font: 'ui', dy: 14, alpha: s.labelsA });
      }
      // edge highlight when an entry is ~0
      if (s.showP > 0.5) for (let i = 0; i < 3; i++) if (s.p[i] < 0.02) { const a = E[(i + 1) % 3], b = E[(i + 2) % 3]; g.line(a, b, { color: 'teal', width: 4.5 }); g.label(NAMES[i] + ' = 0 on this edge', lerpV(a, b, 0.5), { color: 'teal', size: 12, font: 'ui', dy: 16 }); }
    }

    // the white probability point with component drops
    if (s.showP > 0.01) {
      const p = s.p;
      if (s.drops > 0.01) {
        const a = [p[0], 0, 0], b = [p[0], p[1], 0];
        g.line([0, 0, 0], a, { color: 'i', width: 1.6, dashed: true, alpha: s.drops });
        g.line(a, b, { color: 'j', width: 1.6, dashed: true, alpha: s.drops });
        g.line(b, p, { color: 'k', width: 1.6, dashed: true, alpha: s.drops });
      }
      g.point(p, { r: 6.5, color: 'white', alpha: s.showP, stroke: 'bg' });
      if (s.okMark) g.marker.check(p, { dx: 18, dy: -14 });
      if (s.pHandle) g.handle('p', p, (q, st) => { st.p = toSimplex(q); }, { color: 'white', plane: { point: CENTER, normal: [1, 1, 1] } });
    }

    // misconception: two impostors
    if (s.wrong > 0.01) {
      const a = s.wrong, q = N.sub(BAD_HI, N.scale([1, 1, 1], 0.2 / 3));
      g.point(BAD_HI, { r: 6, color: 'bad', alpha: a });
      g.line(BAD_HI, q, { color: 'bad', dashed: true, width: 1.6, alpha: a });
      g.point(q, { r: 3, color: 'bad', alpha: 0.6 * a });
      g.marker.cross(BAD_HI, { dx: 20, dy: -12, size: 20 });
      g.label('Σ = 1.2: above the plane', BAD_HI, { color: 'bad', size: 12, font: 'ui', anchor: 'left', dx: 34, dy: -14, alpha: a });
      g.point(BAD_NEG, { r: 6, color: 'bad', alpha: a });
      g.line(E[0], BAD_NEG, { color: 'bad', dashed: true, width: 1, alpha: 0.5 * a });
      g.line(E[2], BAD_NEG, { color: 'bad', dashed: true, width: 1, alpha: 0.5 * a });
      g.marker.cross(BAD_NEG, { dx: 20, dy: 10, size: 20 });
      g.label('dog = −0.1: outside the triangle', BAD_NEG, { color: 'bad', size: 12, font: 'ui', dy: 32, alpha: a });
    }

    // softmax / temperature point (gold)
    const sm = N.softmax(s.z, s.tempOn || s.satT >= 0 || s.barsMode === 'temp' ? s.T : 1);
    if (s.tempOn) { const tp = tempPath(s.z); g.polyline(tp, { color: 'gold', width: 1.6, alpha: 0.7 }); g.label('T small', tp[0], { color: 'gold', size: 11.5, font: 'ui', dx: 10, dy: -22 }); g.label('T large', CENTER, { color: 'gold', size: 11.5, font: 'ui', dx: 34, dy: -8 }); }
    if (s.showSoft || s.satT >= 0) {
      g.point(sm, { r: 7, color: 'gold' });
      g.label('softmax', sm, { color: 'gold', size: 12, font: 'ui', dx: 0, dy: -18 });
      if (s.stage >= 0) g.marker.ring(sm, { r: 13 });
    }
    if (s.satT >= 0 && s.satT > 0.35) g.label('saturated: pinned in the corner', sm, { color: 'gold', size: 12.5, font: 'ui', anchor: 'left', dy: -36, dx: 16 });
    if (s.target) { g.point(s.target, { r: 5, color: 'hud', alpha: 0.6 }); g.label('target', s.target, { color: 'hud', size: 11, font: 'ui', dy: 14 }); }

    // stochastic matrix: image triangle, columns, a sample point and the fixed point
    if (s.imageT >= 0 || s.showImage) drawStochastic(g, s);

    drawBars(g, s, sm);
    if (s.satT >= 0) drawSqrtD(g, s);
    if (s.tHandle) drawTTrack(g, s);
  }

  function drawStochastic(g, s) {
    const u = s.imageT >= 0 ? s.imageT : 1;
    const cols = [0, 1, 2].map(j => N.col(s.P, j));
    const morph = seg(u, 0.05, 0.45);
    const corners = cols.map((c, j) => lerpV(E[j], c, morph));
    g.polygon(corners, { fill: 'w', alpha: 0.3, stroke: 'w', width: 1.8 });
    cols.forEach((c, j) => { g.point(c, { r: 5.5, color: 'w', alpha: seg(u, 0, 0.15) }); g.label('col' + '₁₂₃'[j], c, { color: 'w', size: 12, font: 'ui', dx: j === 2 ? 26 : -24, dy: j === 2 ? 0 : 10 }); });
    // a sample point and its image, built tip to tail from the columns
    const a = seg(u, 0.5, 0.75);
    if (a > 0.01) {
      const q = s.sample, Pq = N.matVec(s.P, q);
      g.point(q, { r: 5, color: 'white', alpha: a });
      g.label('p', q, { color: 'white', size: 13, dx: 10, dy: -8 });
      // Pp is the p-weighted average of the three column points: spokes to each column, labelled with the weight
      cols.forEach((c, j) => { g.line(Pq, c, { color: 'w', width: 1 + 3 * q[j], alpha: 0.8 * a }); g.label(LA.fmt.fixed(q[j], 2), lerpV(Pq, c, 0.5), { color: 'w', size: 11, font: 'ui', alpha: a }); });
      g.line(q, Pq, { color: 'hud', dashed: true, width: 1.2, alpha: a });
      g.point(Pq, { r: 5.5, color: 'w', alpha: a });
      g.label('Pp', Pq, { color: 'w', size: 13, dx: 14, dy: 10 });
    }
    const b = seg(u, 0.8, 1);
    if (b > 0.01) { const pi = N.stationary(s.P); g.point(pi, { r: 7, color: 'gold', alpha: b }); g.label('fixed point π (next lesson)', pi, { color: 'gold', size: 12, font: 'ui', anchor: 'right', dx: -10, dy: 24, alpha: b }); }
  }

  function drawBars(g, s, sm) {
    if (s.barsMode === 'none' || s.bars < 0.01) return;
    const W = g.size.w, x = W - 176, y = 74, w = 150, h = 92;
    let vals, title, max = 1;
    if (s.barsMode === 'p') { vals = s.p.map(v => v * s.bars); title = 'p: cat / dog / bird'; max = 1.15; }
    else if (s.barsMode === 'soft') {
      const ez = s.z.map(Math.exp), tot = N.sum(ez), st = Math.max(0, s.stage);
      const u1 = seg(st, 0, 1, 'inOut'), u2 = seg(st, 1, 2, 'inOut');
      vals = s.z.map((z, i) => LA.lerp(LA.lerp(z, ez[i], u1), ez[i] / tot, u2)); max = LA.lerp(8.6, 1.15, u2);   // rescale while dividing so the final shares stay readable
      title = st < 1 ? 'scores z → eᶻ (all positive)' : 'eᶻ ÷ ' + LA.fmt(tot) + ' (total height 1)';
    } else { vals = sm; title = 'softmax(z / T)'; max = 1.15; }
    g.bars(vals.map(v => Math.max(0, v)), { screen: true, x, y, w, h, max, colors: COLS, labels: NAMES, values: true, title });
    if (s.barsMode === 'soft' || s.barsMode === 'temp') g.hud('z = ' + fmtP(s.z), x, y + h + 30, { color: 'hud', size: 12 });
    if (s.barsMode === 'temp' && (s.tempOn || s.tHandle)) g.hud('T = ' + LA.fmt.fixed(s.T, 2), x, y + h + 48, { color: 'gold', size: 12.5, bold: true });
  }

  /** small HUD: a dot product of two random d-vectors grows like √d; dividing by √d keeps it near 1 */
  function drawSqrtD(g, s) {
    const a = seg(s.satT, 0.45, 0.7); if (a < 0.01) return;
    const H = g.size.h, pts = [], flat = [];
    for (let d = 1; d <= 512; d *= 2) { pts.push([Math.log2(d), Math.sqrt(d)]); flat.push([Math.log2(d), 1]); }
    g.lineChart([{ points: pts, color: 'bad', width: 2, label: 'typical |q·k| ~ √d' }, { points: flat, color: 'gold', width: 2, label: '÷ √d' }], { x: 20, y: H - 140, w: 200, h: 90, xRange: [0, 9], yRange: [0, 24], title: 'synthetic: score size vs dimension d', xLabel: 'd = 1 … 512 (log scale)' });
  }

  /** a 2D strip at the bottom with a draggable temperature dot (log scale) */
  function drawTTrack(g, s) {
    const W = g.size.w, H = g.size.h;
    g.viewport({ x: 0, y: H - 64, w: W, h: 64 }, { view: '2d', cam: { center: [0.5, 0], halfH: 0.5 }, border: true, bg: 'panel' }, (g2, r) => {
      const P = g2.P, x0 = r.x + 60, x1 = r.x + r.w - 40, yy = r.y + r.h * 0.45;
      const a = P.toWorld(x0, yy), b = P.toWorld(x1, yy);
      g2.line(a, b, { color: 'hud', width: 2 });
      [0.1, 0.25, 0.5, 1, 2, 5].forEach(T => { const sx = x0 + (x1 - x0) * uOf(T); g2.label(String(T), [sx, yy + 16], { screen: true, size: 10.5, color: 'muted', font: 'ui' }); });
      g2.label('T', [r.x + 30, yy], { screen: true, size: 15, color: 'gold', font: 'math' });
      const pos = P.toWorld(x0 + (x1 - x0) * LA.clamp(uOf(s.T), 0, 1), yy);
      g2.point(pos, { r: 6.5, color: 'gold' });
      g2.handle('T', pos, (q, st) => { const sx = P.toScreen(q)[0]; st.T = tOf(LA.clamp((sx - x0) / (x1 - x0), 0, 1)); }, { color: 'gold' });
    });
  }

  /* ---------------- ML tie-in thumbnails (2D) ---------------- */
  function tri2(g2, alpha) {
    g2.polygon(C2, { fill: 'teal', alpha: 0.18 * (alpha === undefined ? 1 : alpha), stroke: 'teal' });
  }
  function renderML(g, s) {
    const u = s.mlT;
    // 1. classifier: three class regions in the input plane, the probe's softmax point on the triangle
    const Wc = [[1.6, 0.2], [-1, 1.2], [-0.6, -1.4]], bc = [0, 0.2, 0];
    const scores = x => Wc.map((w, i) => w[0] * x[0] + w[1] * x[1] + bc[i]);
    const probe = [1.3 * Math.cos(2 * Math.PI * u), 1.1 * Math.sin(2 * Math.PI * u)];
    g.viewport({ x: 0, y: 0, w: 0.5, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 1.6 }, border: true }, (g2, r) => {
      g2.heat2d(x => { const sc = scores(x); return sc.indexOf(Math.max.apply(null, sc)) / 2; }, { x0: -2.5, x1: 2.5, y0: -1.6, y1: 1.6 }, { res: 48, range: [0, 1], ramp: t => t < 0.25 ? C.data : t < 0.75 ? C.dataB : C.dataC, alpha: 0.35, smooth: false });
      g2.point(probe, { r: 6, color: 'white' });
      g2.hud('classifier regions (synthetic)', r.x + 8, r.y + 16, { size: 11.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0.5, y: 0, w: 0.5, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0.1], halfH: 1.1 }, border: true }, (g2, r) => {
      tri2(g2);
      ['A', 'B', 'C'].forEach((nm, i) => g2.label('class ' + nm, C2[i], { color: ['data', 'dataB', 'dataC'][i], size: 11.5, font: 'ui', dy: i === 2 ? 6 : 12, dx: i === 2 ? 40 : 0 }));
      g2.point(to2(N.softmax(scores(probe))), { r: 6, color: 'gold' });
      g2.hud('its softmax on the simplex', r.x + 8, r.y + 16, { size: 11.5, color: 'hud', bold: true });
    });
    // 2. attention: softmax weights over six tokens
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [2.6, 0.75], halfH: 1.15 }, border: true }, (g2, r) => {
      const qk = [0.4, 2.2, 1.1, -0.3, 0.2, 1.6], w = N.softmax(qk, 1 / Math.max(0.05, u));
      g2.bars(w, { origin: [0, 0], width: 0.62, gap: 0.3, height: 1.5, max: 0.6, color: 'gold', labels: ['the', 'cat', 'sat', 'on', 'the', 'mat'], values: true });
      g2.hud('attention weights for one query: a softmax row (sums to 1)', r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
    });
    // 3. Markov: a trail on the triangle converging to the gold fixed point
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0.22], halfH: 1.3 }, border: true }, (g2, r) => {
      tri2(g2);
      ['sunny', 'cloudy', 'rainy'].forEach((nm, i) => g2.label(nm, C2[i], { color: 'muted', size: 11.5, font: 'ui', dy: i === 2 ? 4 : 12, dx: i === 2 ? 32 : 0 }));
      const kk = 12 * u, k0 = Math.floor(kk);
      let p = [0, 0, 1]; const pts = [to2(p)];
      for (let k = 0; k < k0; k++) { p = N.matVec(PW, p); pts.push(to2(p)); }
      const nxt = N.matVec(PW, p), cur = to2(lerpV(p, nxt, kk - k0));
      g2.polyline(pts.concat([cur]), { color: 'hud', width: 1.4 });
      pts.forEach(q => g2.point(q, { r: 2.5, color: 'hud' }));
      g2.point(to2(N.stationary(PW)), { r: 7, color: 'gold' });
      g2.point(cur, { r: 5, color: 'white' });
      g2.hud('Markov chain: π(t+1) = P π(t) walks to the fixed point (p4-05)', r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
    });
  }

  /* ---------------- recap ---------------- */
  function renderRecap(g, s) {
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0.9, 0.12], halfH: 1.15 }, border: true }, (g2, r) => {
      tri2(g2);
      const p = [0.5, 0.3, 0.2];
      g2.point(to2(p), { r: 6, color: 'white' });
      g2.bars(p, { origin: [1.6, -0.7], width: 0.3, gap: 0.12, height: 1.4, max: 1, colors: COLS, labels: NAMES, values: true });
      g2.hud('Triangle of probabilities', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [2.9, 1.0], halfH: 1.4 }, border: true }, (g2, r) => {
      const ez = Z0.map(Math.exp), sm = N.softmax(Z0);
      g2.bars(Z0.map(x => x / 4), { origin: [0, 0], width: 0.32, gap: 0.08, height: 2, max: 1, colors: COLS });
      g2.label('z', [0.55, 0], { color: 'hud', size: 13, dy: 14 });
      g2.label('→ exp →', [1.55, 0.8], { color: 'muted', size: 12, font: 'ui' });
      g2.bars(ez.map(x => x / 8), { origin: [2.1, 0], width: 0.32, gap: 0.08, height: 2, max: 1, colors: COLS });
      g2.label('eᶻ', [2.65, 0], { color: 'hud', size: 13, dy: 14 });
      g2.label('→ ÷ sum →', [3.75, 0.8], { color: 'muted', size: 12, font: 'ui' });
      g2.bars(sm, { origin: [4.3, 0], width: 0.32, gap: 0.08, height: 2, max: 1, colors: COLS });
      g2.label('softmax', [4.85, 0], { color: 'gold', size: 12, font: 'ui', dy: 14 });
      g2.hud('exp, then divide', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0.12], halfH: 1.15 }, border: true }, (g2, r) => {
      tri2(g2);
      const cols = [0, 1, 2].map(j => to2(N.col(PW, j)));
      g2.polygon(cols, { fill: 'w', alpha: 0.3, stroke: 'w' });
      g2.point(to2(N.stationary(PW)), { r: 7, color: 'gold' });
      g2.label('π', to2(N.stationary(PW)), { color: 'gold', size: 14, dx: 12, dy: -10 });
      g2.hud('Triangle into itself ⇒ fixed point, λ = 1', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
  }
})();
