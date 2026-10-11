/* p6-07 — LayerNorm and BatchNorm as geometry: LN = remove the shadow on (1,1,1), then scale onto the circle of
   radius √3 (each vector alone); BN = center and axis-scale the cloud across the batch (the tilt stays). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, fmt = LA.fmt, lerp = LA.lerp, lerpV = LA.lerpV;
  const vS = '<span class="c-v">x</span>';
  const MEAN = [1.5, 0.5, -1.0], COV = [[1.0, 0.6, 0.2], [0.6, 0.8, 0.1], [0.2, 0.1, 0.3]];
  const X = LA.rng.gaussianCloud(607, 40, { mean: MEAN, cov: COV });     // synthetic activations, 40×3
  const SEL = 7, R3 = Math.sqrt(3), ONE = [1, 1, 1];
  const E1 = N.normalize([1, -1, 0]), E2 = N.normalize([1, 1, -2]);         // orthonormal basis of the centering plane
  const ln = x => { const mu = (x[0] + x[1] + x[2]) / 3, u = x.map(v => v - mu), sg = N.norm(u) / R3; return { mu, u, sg, xh: sg > 1e-9 ? u.map(v => v / sg) : [0, 0, 0] }; };
  function colStats(idx) { const mu = [0, 0, 0], sd = [0, 0, 0]; idx.forEach(i => { for (let j = 0; j < 3; j++) mu[j] += X[i][j] / idx.length; }); idx.forEach(i => { for (let j = 0; j < 3; j++) sd[j] += (X[i][j] - mu[j]) ** 2 / idx.length; }); return { mu, sd: sd.map(v => Math.sqrt(Math.max(v, 1e-12))) }; }
  const ALL = X.map((_, i) => i);
  const BS = colStats(ALL);
  const sqrtCov = C => { const e = N.eigSym(C); return N.fromCols(e.vectors.map((q, i) => N.scale(q, Math.sqrt(Math.max(e.values[i], 0))))); };
  const covOf = P => N.cov(P, { ddof: 0 });
  const BATCH = [0, 1, 2].map(k => LA.rng.shuffle(LA.rng(607 + k), ALL).slice(0, 4));
  const WH = N.whiten(X);                                                   // for the "not whitening" contrast
  const STRIP = (function () { const r = LA.rng(612), v = []; for (let i = 0; i < 12; i++) v.push(+(0.8 + LA.rng.gauss(r) * 0.9).toFixed(2)); return v; })();
  const STRIP_LN = (function () { const mu = N.sum(STRIP) / 12, u = STRIP.map(v => v - mu), sg = N.norm(u) / Math.sqrt(12); return u.map(v => v / sg); })();

  /** where a point is at phase p ∈ [0,3] of LayerNorm (0 raw · 1 centered · 2 scaled · 3 γ⊙x̂ + β) */
  function lnAt(x, p, gamma, beta) {
    const L = ln(x);
    if (p <= 1) return x.map(v => v - p * L.mu);
    if (p <= 2) { const k = lerp(1, 1 / L.sg, p - 1); return L.u.map(v => v * k); }
    return L.xh.map((v, j) => lerp(v, gamma[j] * v + beta[j], p - 2));
  }
  /** BatchNorm phase p ∈ [0,3] with given column statistics */
  function bnAt(x, p, st, gamma, beta) {
    const c = x.map((v, j) => v - Math.min(1, p) * st.mu[j]);
    if (p <= 1) return c;
    const xh = x.map((v, j) => (v - st.mu[j]) / st.sd[j]);
    if (p <= 2) return c.map((v, j) => v * lerp(1, 1 / st.sd[j], p - 1));
    return xh.map((v, j) => lerp(v, gamma[j] * v + beta[j], p - 2));
  }
  const circlePts = (r, n) => { const P = []; for (let k = 0; k <= (n || 64); k++) { const a = k / (n || 64) * 2 * Math.PI; P.push(N.add(N.scale(E1, r * Math.cos(a)), N.scale(E2, r * Math.sin(a)))); } return P; };
  const cam3 = o => Object.assign(LA.cam.defaults3d(), { target: [0.4, 0.1, -0.05], dist: 11.2, yaw: -0.45, pitch: 0.3, fov: 36 }, o || {});

  LA.registerLesson({
    id: 'p6-07-normalization-layers',
    part: 6,
    title: 'LayerNorm and BatchNorm as geometry',
    subtitle: 'Project onto a plane, then onto a circle — or center and axis-scale the whole cloud.',
    minutes: 10,
    prereqs: ['p3-03-projections', 'p4-09-covariance', 'p1-05-norms-distance', 'p4-11-whitening-mahalanobis', 'p5-03-gradient-descent'],
    tags: ['extension', 'capstone'],
    badge: { drawnIn: '3D, exact for 3 features', realSize: 'LayerNorm over d = 4096 per token; BatchNorm over a batch of 256 × 56×56 positions per channel', transfers: 'LN = projection onto the plane ⟂ 1, then scaling to length √d; BN = per-axis centering and scaling — exact for any d' },
    cantShow: ['The sphere of radius √d inside the (d − 1)-dimensional centering hyperplane for d > 3.', 'Running statistics used at inference time.', 'RMSNorm (LayerNorm without the centering step) beyond one sentence.'],
    terms: [
      { term: 'LayerNorm', def: 'Per vector: subtract its own mean, divide by its own spread, then scale by γ and shift by β.' },
      { term: 'BatchNorm', def: 'Per feature: subtract the batch mean and divide by the batch spread.' },
      { term: 'all-ones direction', def: 'The direction 1 = (1, 1, 1), along which a vector’s mean lives.' },
      { term: 'centering plane', def: 'The plane through the origin perpendicular to 1, where mean-zero vectors live.' },
      { term: 'γ and β', def: 'The layer’s learned per-feature scale and shift, applied after normalizing.' },
      { term: 'batch statistics', def: 'Means and spreads computed across the samples of a batch.' },
    ],
    atlas: ['normalization', 'optimization'],
    thumbnail: 4,

    state() {
      return {
        view: '3d', cam: cam3(), scene: 'cloud', panel: 'cloud',
        v: X[SEL].slice(), mode: 'ln', bn: false, phase: 0, gamma: [1, 1, 1], beta: [0, 0, 0], batch: 40, showCov: false,
        showArrows: true, showEll: true, others: 1, oneVec: false, allLN: false, slide: 0, grow: 0, shadow: 0,
        batchK: -1, whiteGhost: 0, guessLN: [0, 0, 0], cross: false, check: false, showPlane: false, showCircle: false, showLine: false,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'ml') return renderML(g, s);
      if (s.scene === 'recap') return renderRecap(g, s);
      scene3d(g, s, ctx);
    },

    math(s, tex) {
      const H = str => tex.text(str, { html: true });
      const T = (v, c) => tex.text('(' + v.map(a => fmt(a)).join(', ') + ')', { color: c });
      const L = ln(s.v);
      switch (s.panel) {
        case 'cloud': return tex.block(H('X ∈ ℝ<sup>40×3</sup>'), tex.eq(tex.text('mean'), T(BS.mu)));
        case 'shadow': return tex.lines(tex.eq(tex.sym('μ'), tex.op('='), H('(1ᵀ<span class="c-v">x</span>)/3 ='), tex.text(fmt(L.mu), { color: 'mag' })), tex.eq(tex.text('shadow'), tex.op('='), H('μ·1 ='), T([L.mu, L.mu, L.mu], 'mag')));
        case 'center': return tex.lines(H('u = <span class="c-v">x</span> − μ1 = (I − 11ᵀ/3)<span class="c-v">x</span>'), tex.eq(tex.text('u'), tex.op('='), T(L.u, 'teal'), tex.text('· mean 0')));
        case 'scale': return tex.lines(tex.eq(H('σ = ‖u‖/√3 ='), tex.text(fmt(L.sg))), tex.eq(H('x̂ = u/σ ='), T(L.xh, 'gold')), H('‖x̂‖ = √3 ≈ 1.73'));
        case 'all': return tex.lines(H('1ᵀx̂ = 0 and ‖x̂‖ = √3 for every row'));
        case 'gb': return tex.lines(H('y = γ ⊙ x̂ + β'), tex.eq(tex.text('γ'), T(s.gamma, 'b'), tex.text('β'), T(s.beta, 'b')));
        case 'mis': return tex.lines(H('LN: per <b>row</b> (one vector, its 3 features)'), H('BN: per <b>column</b> (one feature, across the batch)'));
        case 'bnc': return tex.lines(H('μ<sub>j</sub> = mean over rows'), tex.eq(tex.text('μ'), T(BS.mu)), H('x<sub>j</sub> ← x<sub>j</sub> − μ<sub>j</sub>: covariance unchanged'));
        case 'bns': { const C = covOf(X.map(x => bnAt(x, Math.min(2, s.phase), BS, s.gamma, s.beta))); return tex.lines(H('x̂<sub>j</sub> = (x<sub>j</sub> − μ<sub>j</sub>)/σ<sub>j</sub>'), tex.eq(tex.text('diag(Cov)'), T(N.diagOf(C))), tex.eq(tex.text('off-diagonal'), T([C[0][1], C[0][2], C[1][2]], 'gold'))); }
        case 'batch': { const idx = s.batchK >= 0 ? BATCH[s.batchK] : ALL; const st = colStats(idx); return tex.lines(tex.eq(tex.text('batch size'), tex.text(String(idx.length), { color: 'gold' })), tex.eq(tex.text('batch μ'), T(st.mu)), tex.eq(tex.text('full-data μ'), T(BS.mu))); }
        case 'ml': return tex.lines(H('x̂ = (x − μ1)/σ ∈ ℝ<sup>4096</sup>'), H('‖x̂‖ = √d'));
        case 'play': {
          const y = s.bn ? bnAt(s.v, s.phase, BS, s.gamma, s.beta) : lnAt(s.v, s.phase, s.gamma, s.beta);
          return tex.lines(tex.eq(tex.text(s.bn ? 'BatchNorm' : 'LayerNorm'), tex.text('phase ' + fmt(s.phase))), tex.eq(tex.sym('x', { color: 'v' }), tex.op('='), T(s.v, 'v'), tex.op('→'), T(y, 'gold')), tex.eq(tex.text('mean'), tex.text(fmt((s.v[0] + s.v[1] + s.v[2]) / 3)), tex.text('‖x‖'), tex.text(fmt(N.norm(s.v)))));
        }
        case 'cam': return tex.lines(H('look along (1, 1, 1): the plane faces you'));
        case 'entry': return tex.lines(tex.eq(tex.text('your LN(1, 2, 3)'), T(s.guessLN, 'w')), H('μ = ?, deviations = ?, σ = ‖u‖/√3'));
        default: return '';
      }
    },

    steps: [
      { title: 'Why normalize', duration: 3.5,
        narration: 'Activations drift: after a few layers a batch of 3-feature vectors sits off-center with a long, tilted spread. p5-03 showed what that does to training: an elongated bowl and zig-zags. Normalization layers put the cloud back where gradient descent likes it.',
        setup(s) { s.scene = 'cloud'; s.panel = 'cloud'; s.view = '3d'; s.cam = cam3(); s.mode = 'ln'; s.phase = 0; s.showArrows = true; s.showEll = true; s.others = 1; s.oneVec = false; s.allLN = false; s.showPlane = false; s.showCircle = false; s.showLine = false; s.shadow = 0; s.slide = 0; s.grow = 0; s.gamma = [1, 1, 1]; s.beta = [0, 0, 0]; s.v = X[SEL].slice(); s.cross = false; s.check = false; s.batchK = -1; s.whiteGhost = 0; s.showCov = false; },
        at(s, t) { LA.cam.tween(s, { yaw: -0.75 }, { yaw: -0.45 }, seg(t, 0, 1, 'linear')); } },

      { title: 'LayerNorm, one vector: its mean is a shadow', duration: 4,
        narration: 'Take one vector ' + vS + '. Its mean μ = (x₁ + x₂ + x₃)/3 is not a new idea: it is the shadow of x onto the <b>all-ones direction</b> 1 = (1, 1, 1) (p3-03), scaled. The magenta shadow μ·1 is “how much of x is just a common level”.',
        setup(s) { s.panel = 'shadow'; s.showArrows = false; s.showEll = false; s.oneVec = true; s.showLine = true; s.shadow = 0; },
        at(s, t) { s.others = 1 - 0.75 * seg(t, 0, 0.3); s.shadow = seg(t, 0.3, 0.8); } },

      { title: 'Subtract the mean: project onto a plane', duration: 4,
        narration: 'Subtracting the mean removes the shadow: u = x − μ1 is what is left, and it lies in the teal plane perpendicular to 1 through the origin, the <b>centering plane</b>. This is the projection I − 11ᵀ/3 from p3-03, applied to every vector in the layer.',
        setup(s) { s.panel = 'center'; s.others = 0.25; s.shadow = 1; s.showPlane = true; s.phase = 0; },
        at(s, t) { s.phase = seg(t, 0.25, 0.9); } },

      { title: 'Divide by the spread: onto a circle', duration: 4,
        narration: 'The spread of the three entries is σ = ‖u‖/√3. Dividing by it scales u to length √3 (p1-05): the tip lands on a gold circle of radius √3 in the plane. Two numbers are gone, the vector’s level and its scale. Its direction within the plane remains.',
        setup(s) { s.panel = 'scale'; s.phase = 1; s.shadow = 0; s.showCircle = true; },
        at(s, t) { s.phase = 1 + seg(t, 0.2, 0.85); } },

      { title: 'Every vector lands on the same circle', duration: 4,
        narration: 'Do it to all forty. Every tip ends on that one circle: a whole cloud collapsed to directions in a plane. <b>LayerNorm</b> never looks at the other samples; each vector is normalized using only its own three numbers.',
        setup(s) { s.panel = 'all'; s.phase = 0; s.allLN = true; s.others = 1; s.oneVec = false; s.showLine = true; },
        at(s, t) { s.phase = 2 * seg(t, 0.05, 0.9); } },

      { title: 'γ and β', duration: 3.5,
        narration: 'Finally the layer’s own learned numbers, <b>γ and β</b>: multiply each feature by γⱼ and add βⱼ. The axis scales stretch the circle into an ellipse that tilts out of the plane; the shift slides it away from the origin. The network can undo the normalization where it needs to, but starts from a clean circle.',
        setup(s) { s.panel = 'gb'; s.phase = 3; s.allLN = true; s.gamma = [1, 1, 1]; s.beta = [0, 0, 0]; },
        at(s, t) { const u = seg(t, 0.1, 0.85); s.gamma = lerpV([1, 1, 1], [1.6, 1, 0.6], u); s.beta = lerpV([0, 0, 0], [0.5, 0, 0], u); } },

      { title: 'LayerNorm does not look across the batch', duration: 3.5, misconception: true,
        narration: '“LayerNorm makes each feature mean zero across the batch.” No, that is <b>BatchNorm</b>. LayerNorm works inside one vector, across its features. Hide the other 39 points: the result for this vector does not change at all. The batch is irrelevant to it.',
        setup(s) { s.panel = 'mis'; s.gamma = [1, 1, 1]; s.beta = [0, 0, 0]; s.phase = 2; s.allLN = true; s.oneVec = true; s.others = 1; s.cross = true; s.check = false; },
        at(s, t) { s.others = 1 - seg(t, 0.1, 0.5); s.check = t > 0.55; } },

      { title: 'BatchNorm: center the cloud', duration: 3.5,
        narration: 'BatchNorm goes the other way: for each feature, subtract the mean over the batch. The whole cloud slides so its center sits at the origin: the centering step of p4-09, feature by feature. The covariance heatmap does not change.',
        setup(s) { s.panel = 'bnc'; s.mode = 'bn'; s.allLN = false; s.oneVec = false; s.cross = false; s.check = false; s.others = 1; s.showPlane = false; s.showCircle = false; s.showLine = false; s.showEll = true; s.showCov = true; s.phase = 0; s.cam = cam3({ target: [0.2, 0, 0], dist: 10.8 }); },
        at(s, t) { s.phase = seg(t, 0.1, 0.85); } },

      { title: 'Scale each axis', duration: 4.5,
        narration: 'Then divide each feature by its batch spread σⱼ: an axis-aligned scaling. The cloud now has unit spread along x₁, x₂ and x₃, but it stays tilted. Correlations survive: BatchNorm is not whitening (p4-11), which would make the cloud round (the dim ghost).',
        setup(s) { s.panel = 'bns'; s.phase = 1; s.whiteGhost = 0; },
        at(s, t) { s.phase = 1 + seg(t, 0.05, 0.6); s.whiteGhost = seg(t, 0.65, 1); } },

      { title: 'Batch size matters', duration: 4,
        narration: 'BatchNorm’s means and spreads come from the batch, its <b>batch statistics</b>. With four samples they are noisy: redraw the batch and the normalized cloud jumps. With forty it is steady. LayerNorm has no such dependence, one reason transformers use it.',
        setup(s) { s.panel = 'batch'; s.phase = 2; s.whiteGhost = 0; s.batchK = 0; },
        at(s, t) { s.batchK = t < 0.75 ? Math.min(2, Math.floor(t / 0.25)) : -1; s.batch = s.batchK >= 0 ? 4 : 40; } },

      { title: 'Why ML cares: rounder bowls', duration: 4, mlTie: true,
        narration: 'Transformer blocks apply LayerNorm to each token’s d features (d = 4096 in GPT-3 6.7B): the same two moves, ending on a sphere of radius √d inside a hyperplane. RMSNorm, its cousin, skips the centering. BatchNorm sits in CNNs, per channel. Both steady the scale: rounder loss bowls (p5-03), tanh kept in its live zone (p5-06). The picture stops at 3. The formula does not.',
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.view = '2d'; s.cam = LA.cam.defaults2d(); },
        at(s, t) { } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'LayerNorm: remove the shadow on (1, 1, 1), a projection, then scale onto a circle of radius √d; each vector alone. BatchNorm: center and axis-scale the cloud across the batch; the tilt stays. Both are projections and scalings you already drew.',
        cards: ['<b>LN: project ⟂ 1, scale to √d.</b> Each vector lands on the circle in the centering plane.', '<b>BN: center and axis-scale across the batch.</b> The cloud gets unit spread per axis; its tilt stays.', '<b>Rows vs columns.</b> LN normalizes a row of X; BN normalizes a column.'],
        setup(s) { s.scene = 'recap'; s.panel = 'none'; s.view = '2d'; s.cam = LA.cam.defaults2d(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the yellow vector’s tip, scrub the phase, and switch between LayerNorm and BatchNorm. Orbit to look down the gray (1, 1, 1) line.</p>',
      setup(s) { s.scene = 'play'; s.panel = 'play'; s.view = '3d'; s.cam = cam3(); s.bn = false; s.mode = 'ln'; s.phase = 2; s.allLN = true; s.oneVec = true; s.others = 1; s.showPlane = true; s.showCircle = true; s.showLine = true; s.showEll = false; s.showArrows = false; s.v = X[SEL].slice(); },
      controls: [
        { type: 'preset', label: 'Presets', items: [
          { label: 'LN one vector', apply(s) { s.bn = false; s.allLN = false; s.oneVec = true; s.others = 0.25; s.showPlane = true; s.showCircle = true; s.showLine = true; s.showEll = false; s.whiteGhost = 0; } },
          { label: 'LN all', apply(s) { s.bn = false; s.allLN = true; s.oneVec = true; s.others = 1; s.showPlane = true; s.showCircle = true; s.showLine = true; s.showEll = false; s.whiteGhost = 0; } },
          { label: 'BN', apply(s) { s.bn = true; s.allLN = false; s.oneVec = true; s.others = 1; s.showPlane = false; s.showCircle = false; s.showLine = false; s.showEll = true; s.whiteGhost = 0; } },
          { label: 'Compare with whitening', apply(s) { s.bn = true; s.allLN = false; s.oneVec = false; s.others = 1; s.showPlane = false; s.showCircle = false; s.showLine = false; s.showEll = true; s.whiteGhost = 1; s.phase = 2; } },
        ] },
        { type: 'slider', bind: 'phase', min: 0, max: 3, step: 0.01, label: 'Step: raw → centered → scaled → γβ' },
        { type: 'slider', bind: 'gamma[0]', min: 0.5, max: 2, step: 0.01, label: 'γ₁', color: 'b' },
        { type: 'slider', bind: 'beta[0]', min: -1, max: 1, step: 0.01, label: 'β₁', color: 'b' },
        { type: 'slider', bind: 'batch', min: 4, max: 40, step: 1, label: 'Batch size (BN)' },
        { type: 'toggle', bind: 'showCov', label: 'Covariance heatmap' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the yellow vector so that LayerNorm leaves it unchanged.',
        setup(s) { s.scene = 'play'; s.panel = 'play'; s.view = '3d'; s.cam = cam3(); s.bn = false; s.phase = 2; s.allLN = false; s.oneVec = true; s.others = 0.2; s.showPlane = true; s.showCircle = true; s.showLine = true; s.showEll = false; s.showArrows = false; s.v = [1.5, 0.5, -1.0]; },
        judge(s) { const m = (s.v[0] + s.v[1] + s.v[2]) / 3, n = N.norm(s.v); const a = LA.judge.scalar(m, 0, LA.TOL.scalar), b = LA.judge.scalar(n, R3, 2 * LA.TOL.unitNorm); return LA.judge.all(a, b); },
        hint: 'It must already be mean-zero (in the teal plane) and have length √3 (on the gold circle).',
        solution(s) { s.v = [1.2247, -1.2247, 0]; } },
      { kind: 'manipulate',
        prompt: 'Orbit the camera to look straight down the (1, 1, 1) direction so the centering plane is face-on and the gold circle looks round.',
        setup(s) { s.scene = 'play'; s.panel = 'cam'; s.view = '3d'; s.cam = cam3(); s.bn = false; s.phase = 2; s.allLN = true; s.oneVec = true; s.others = 1; s.showPlane = true; s.showCircle = true; s.showLine = true; s.showEll = false; s.showArrows = false; s.v = X[SEL].slice(); },
        judge(s) { return LA.judge.lookAlong(s.cam, ONE, 2 * LA.TOL.angleDeg); },
        hint: 'Rotate until the gray diagonal line shrinks to a point.',
        solution(s) { s.cam = LA.cam.lookAlong(s.cam, ONE); s.cam.auto = false; } },
      { kind: 'entry',
        prompt: 'Type LayerNorm of x = (1, 2, 3) with γ = 1, β = 0. Your vector is drawn purple.',
        bind: 'guessLN', answer: [-1.225, 0, 1.225], tol: LA.TOL.entry, label: 'LN(x)', colors: 'w',
        hint: 'μ = 2; deviations (−1, 0, 1); σ = √(2/3) ≈ 0.82.',
        setup(s) { s.scene = 'entry'; s.panel = 'entry'; s.view = '3d'; s.cam = cam3({ target: [0.3, 0.4, 0.7], dist: 12 }); s.guessLN = [0, 0, 0]; s.showArrows = false; s.showPlane = true; s.showCircle = true; s.showLine = true; s.showEll = false; s.allLN = false; s.oneVec = false; s.others = 0; } },
    ],
  });

  /* ---------------- 3D scene ---------------- */
  function covInset(g, C, label) {
    const W = g.size.w, H = g.size.h, w = 150;
    g.viewport({ x: W - w - 12, y: H - w - 34, w, h: w + 22 }, { view: '2d', cam: { center: [1.5, -1.35], halfH: 1.85 }, bg: 'panel' }, (g2, r) => {
      const grid = C.map(row => row.map(v => v));
      g2.heat2d(grid, { x0: 0.1, x1: 2.9, y0: -2.9, y1: -0.1 }, { range: [-1.2, 1.2], ramp: t => LA.color.diverge(2 * t - 1), smooth: false, alpha: 1 });
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) g2.label(fmt(C[i][j]), [0.1 + (j + 0.5) * 0.933, -0.1 - (i + 0.5) * 0.933], { size: 11, color: 'text', bg: false, font: 'ui' });
      g2.label(label, [r.x + r.w / 2, r.y + 12], { screen: true, size: 11.5, color: 'muted', font: 'ui' });
    });
  }
  function scene3d(g, s, ctx) {
    const play = s.scene === 'play', bnMode = play ? s.bn : s.mode === 'bn';
    g.axes({ length: 2.8, names: ['x₁', 'x₂', 'x₃'] });
    if (s.showLine) { g.lineThrough([0, 0, 0], ONE, { color: 'residual', len: 3.4, width: 1.3, alpha: 0.8 }); g.label('1 = (1, 1, 1)', [1.75, 1.75, 1.75], { color: 'muted', size: 12, anchor: 'left', dx: 6 }); }
    if (s.panel === 'cloud') g.hud('40 synthetic 3-feature activations', 16, 24, { color: 'muted' });
    if (s.showPlane) g.planes([{ point: [0, 0, 0], normal: ONE, extent: 2.6, color: 'teal', alpha: 0.16 }]);
    if (s.showCircle) g.polyline(circlePts(R3), { color: 'gold', width: 2.2 });
    if (s.scene === 'entry') return entry3d(g, s);
    // batch statistics in use (batch-size step and playground)
    let st = BS, bIdx = null;
    if (s.batchK >= 0) { bIdx = BATCH[s.batchK]; st = colStats(bIdx); }
    else if (play && bnMode && s.batch < 40) { bIdx = ALL.slice(0, Math.round(s.batch)); st = colStats(bIdx); }
    // the cloud
    let P;
    if (bnMode) P = X.map(x => bnAt(x, s.phase, st, s.gamma, s.beta));
    else if (s.allLN) P = X.map(x => lnAt(x, s.phase, s.gamma, s.beta));
    else P = X.map(x => x.slice());
    const aPts = s.others;
    if (aPts > 0.02) {
      if (s.allLN && s.phase > 0.02 && !bnMode) X.forEach(x => g.point(x, { r: 2.2, color: 'data', alpha: 0.18 * aPts }));   // ghosts of the raw cloud
      if (s.showArrows) P.forEach(p => g.line([0, 0, 0], p, { color: 'data', width: 0.8, alpha: 0.18 }));
      g.points(P, { r: 3.4, color: 'data', alpha: 0.9 * aPts });
      if (bIdx && bIdx.length <= 6) bIdx.forEach(i => g.point(P[i], { r: 6, color: 'rgba(0,0,0,0)', stroke: 'gold', strokeWidth: 1.6 }));
    }
    if (s.showEll && aPts > 0.02) {
      const C = covOf(P), mu = N.mean(P);
      g.ellipsoid(sqrtCov(C), { center: mu, color: 'teal', alpha: 0.75, meridians: 4 });
    }
    if (s.whiteGhost > 0.01) { const Wp = WH.Xw; g.points(Wp, { r: 2.6, color: 'muted', alpha: 0.45 * s.whiteGhost }); g.sphere([0, 0, 0], 1, { color: 'muted', alpha: 0.35 * s.whiteGhost, meridians: 3 }); g.label('whitened (round): not what BN does', [0, 0, 1.25], { color: 'muted', size: 12, alpha: s.whiteGhost }); }
    // the vector under study
    if (s.oneVec) {
      const x = s.v, L = ln(x);
      const y = bnMode ? bnAt(x, s.phase, st, s.gamma, s.beta) : lnAt(x, s.phase, s.gamma, s.beta);
      if (s.shadow > 0.01 && !bnMode) {
        const foot = [L.mu, L.mu, L.mu].map(v => v * s.shadow);
        g.vector([0, 0, 0], [L.mu, L.mu, L.mu], { color: 'mag', width: 3, label: 'μ·1', alpha: s.shadow });
        g.line(x, [L.mu, L.mu, L.mu], { color: 'residual', dashed: true, width: 1.3, alpha: s.shadow });
        void foot;
      }
      if (s.phase > 0.02 && !(s.allLN && s.others < 0.05 && s.check)) g.vector([0, 0, 0], x, { color: 'v', width: 1.6, alpha: 0.3 });
      g.vector([0, 0, 0], y, { color: 'v', width: 3.2, label: bnMode ? (s.phase > 0.02 ? 'BN(x)' : 'x') : (s.phase <= 0.02 ? 'x' : s.phase <= 1.02 ? 'u' : s.phase <= 2.02 ? 'x̂ = LN(x)' : 'y') });
      if (!bnMode && s.phase > 0.02 && s.phase < 1) g.line(y, N.add(y, [L.mu, L.mu, L.mu].map(v => v * (1 - s.phase))), { color: 'residual', dashed: true, width: 1.2 });
      if (s.panel === 'scale' && s.phase > 1.9) { g.dim([0, 0, 0], y, { label: '√3', offset: 16, color: 'gold' }); }
      if (s.check) { g.marker.check(y, { dx: 22, dy: -18, size: 20 }); g.label('unchanged without the batch', y, { dx: -40, dy: 44, size: 12.5, color: 'ok', anchor: 'right' }); }
      if (play || s.scene === 'check') g.handle('v', x, (p, stt) => { stt.v = [p[0], p[1], p[2]]; }, { color: 'v', plane: { point: [0, 0, 0], normal: ONE } });
    }
    // γ, β annotations
    if (s.panel === 'gb') {
      g.polyline(circlePts(R3).map(p => p.map((v, j) => s.gamma[j] * v + s.beta[j])), { color: 'b', width: 2.2 });
      g.vector([0, 0, 0], s.beta, { color: 'b', width: 3, label: 'β' });
      g.hud('γ = (' + s.gamma.map(v => fmt(v)).join(', ') + ')  stretches the axes', 16, 24, { color: 'b' });
    }
    if (s.panel === 'all') g.hud('each vector on its own', 16, 24, { color: 'gold' });
    if (s.panel === 'batch') g.hud(s.batchK >= 0 ? 'batch of 4 (draw ' + (s.batchK + 1) + ' of 3): statistics from the ringed points' : 'batch of 40: steady', 16, 24, { color: s.batchK >= 0 ? 'gold' : 'ok' });
    if (s.panel === 'bnc' || s.panel === 'bns') g.hud(s.panel === 'bnc' ? 'BatchNorm step 1: subtract the batch mean of each feature' : 'step 2: divide each feature by its batch spread', 16, 24, { color: 'hud' });
    if (s.cross) {
      g.hud('“across the batch”', 44, 26, { color: 'bad', size: 15 });
      g.marker.cross([24, 26], { screen: true, size: 20 });
      rowsVsCols(g, s, true);
    }
    if (s.showCov || s.panel === 'bnc' || s.panel === 'bns') covInset(g, covOf(P), 'Cov of the cloud (3×3)');
    if (play && s.panel !== 'cam') g.hud(bnMode ? 'BatchNorm' + (s.batch < 40 ? ' · batch of ' + Math.round(s.batch) : '') : 'LayerNorm', 16, 24, { color: 'hud' });
    if (s.panel === 'cam') g.hud('look down the gray line until it shrinks to a point', 16, 24, { color: 'muted' });
  }
  function rowsVsCols(g, s, crossCol) {
    const W = g.size.w, H = g.size.h, w = 110, h = 200;
    g.viewport({ x: W - w - 14, y: H - h - 40, w, h: h + 26 }, { view: '2d', cam: { center: [3.4, -10.2], halfH: 11.6 }, bg: 'panel' }, (g2, r) => {
      const mn = -2.5, mx = 3.5;
      g2.heat2d(X, { x0: 0, x1: 5.4, y0: -20, y1: 0 }, { range: [mn, mx], ramp: t => LA.color.heat(t), smooth: false, alpha: 0.9 });
      g2.polyline([[0, -7.5], [5.4, -7.5], [5.4, -8], [0, -8]], { color: 'v', width: 2, closed: true });
      g2.vector([0.2, -7.75 + 1.4], [5.2, -7.75 + 1.4], { color: 'ok', width: 2, head: 7, label: 'LN' });
      g2.vector([6.2, -1], [6.2, -19], { color: crossCol ? 'bad' : 'muted', width: 2, head: 7 });
      if (crossCol) g2.marker.cross([6.2, -10], { size: 14 });
      g2.label('X (40×3)', [r.x + r.w / 2, r.y + 12], { screen: true, size: 10.5, color: 'muted', font: 'ui' });
    });
  }
  function entry3d(g, s) {
    const x = [1, 2, 3], L = ln(x), gs = s.guessLN;
    g.vector([0, 0, 0], x, { color: 'v', width: 2.2, label: 'x = (1, 2, 3)', alpha: 0.8 });
    if (N.norm(gs) > 1e-6) g.vector([0, 0, 0], gs, { color: 'w', width: 3.2, label: 'yours' });
    const ok = N.norm(N.sub(gs, L.xh)) < 0.08;
    if (ok) g.marker.check(gs, { dx: 20, dy: -16, size: 20 });
    g.hud('LN(x) must end on the gold circle, inside the teal plane', 16, 24, { color: 'muted' });
  }

  /* ---------------- ML tie-in and recap ---------------- */
  function stripRow(g, vals, x, y, w, title, color) {
    const n = vals.length, cw = w / n, m = Math.max(...vals.map(Math.abs));
    g.strip(vals, { x, y, w, h: 28, range: [-m, m], ramp: t => LA.color.diverge(2 * t - 1), title });
    vals.forEach((v, i) => g.hud(fmt(v), x + (i + 0.5) * cw, y + 14, { anchor: 'center', size: 10.5, color: 'text', bg: false }));
    void color;
  }
  function renderML(g, s) {
    const W = g.size.w, H = g.size.h, x0 = 24, w = W - 48;
    const mu = N.sum(STRIP) / 12;
    stripRow(g, STRIP, x0, 60, w, 'a synthetic 12-feature activation before LN (mean ' + fmt(mu) + ', norm ' + fmt(N.norm(STRIP)) + ')');
    stripRow(g, STRIP_LN, x0, 130, w, 'after LN: mean 0, norm √12 ≈ ' + fmt(Math.sqrt(12)));
    g.hud('The picture stops at 3. The formula does not.', x0, 196, { color: 'gold', size: 14 });
    // p5-03 thumbnail: elongated bowl → round bowl
    const card = (i, title, k) => g.viewport({ x: 0.04 + i * 0.47, y: 0.36, w: 0.44, h: 0.5 }, { view: '2d', cam: { center: [0, 0], halfH: 2.2 }, bg: 'panel' }, (g2, r) => {
      g2.contours(p => p[0] * p[0] / (k * k) + p[1] * p[1], [0.25, 0.75, 1.5, 2.5, 3.6], { x0: -3, x1: 3, y0: -2.2, y1: 2.2 }, { res: 48, color: 'teal', width: 1.4 });
      const path = []; let q = k < 1 ? [-0.8, 1.7] : [-1.4, 1.3];
      for (let it = 0; it < 14; it++) { path.push(q.slice()); const gr = [2 * q[0] / (k * k), 2 * q[1]]; q = [q[0] - 0.18 * gr[0], q[1] - 0.18 * gr[1]]; }
      g2.polyline(path, { color: 'b', width: 2 }); g2.points(path, { r: 2.5, color: 'b' });
      g2.label(title, [r.x + 10, r.y + 14], { screen: true, anchor: 'left', size: 12.5, color: 'text', font: 'ui', bold: true });
    });
    card(0, 'unnormalized: long bowl, zig-zag', 0.45);
    card(1, 'normalized: round bowl', 1);
    g.hud('see p6-08: standardizing the input is the same move', W - 24, H - 24, { anchor: 'right', color: 'muted', size: 12.5 });
  }
  function renderRecap(g, s) {
    const card = (i, vopts, fn, text) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, Object.assign({ bg: 'panel' }, vopts), (g2, r) => { fn(g2, r); g2.label(text, [r.x + 12, r.y + 16], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true }); });
    card(0, { view: '3d', cam: cam3({ dist: 8.5, target: [0.2, 0.1, 0], yaw: -0.45, pitch: 0.45 }) }, (g3) => {
      g3.lineThrough([0, 0, 0], ONE, { color: 'residual', len: 3, width: 1.2 });
      g3.planes([{ point: [0, 0, 0], normal: ONE, extent: 2.3, color: 'teal', alpha: 0.16 }]);
      g3.polyline(circlePts(R3), { color: 'gold', width: 2 });
      const x = X[SEL], L = ln(x);
      g3.vector([0, 0, 0], x, { color: 'v', width: 1.6, alpha: 0.4, label: 'x' });
      g3.vector([0, 0, 0], L.xh, { color: 'v', width: 3, label: 'LN(x)' });
    }, '1 · LN: project ⟂ 1, then scale to √3');
    card(1, { view: '3d', cam: cam3({ dist: 9, target: [0.6, 0.2, -0.4] }) }, (g3) => {
      g3.points(X, { r: 2.2, color: 'data', alpha: 0.3 });
      const P = X.map(x => bnAt(x, 2, BS, [1, 1, 1], [0, 0, 0]));
      g3.points(P, { r: 2.8, color: 'data' });
      g3.ellipsoid(sqrtCov(covOf(P)), { color: 'teal', alpha: 0.7, meridians: 3 });
      g3.axes({ length: 2.2, labels: false });
    }, '2 · BN: center and axis-scale; the tilt stays');
    card(2, { view: '2d', cam: { center: [0, 0], halfH: 1.4 } }, (g2, r) => {
      const xL = g2.P.toWorld(r.x + 40, 0)[0];
      g2.heat2d(X.slice(0, 12), { x0: xL, x1: xL + 0.9, y0: -1.15, y1: 0.95 }, { range: [-2.5, 3.5], ramp: t => LA.color.heat(t), smooth: false, alpha: 0.9 });
      g2.vector([xL + 0.05, 0.6], [xL + 0.85, 0.6], { color: 'ok', width: 2, head: 7, label: 'LN: a row' });
      g2.vector([xL + 1.15, 0.9], [xL + 1.15, -1.1], { color: 'b', width: 2, head: 7, label: 'BN: a column' });
      g2.label('rows = samples, columns = features', [r.x + r.w - 14, r.y + r.h / 2], { screen: true, anchor: 'right', size: 13, color: 'muted', font: 'ui' });
    }, '3 · rows vs columns');
  }
})();
