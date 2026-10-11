/* p4-11 — Whitening, Mahalanobis distance, anomaly detection (rings not circles; rotate → rescale → rotate back = Σ^−1/2). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpM = LA.lerpM, C = LA.color;
  const HONEST = 'The picture stops at 3. The formula does not.';
  const I2 = () => [[1, 0], [0, 1]];
  const R2 = Math.SQRT1_2;
  const P1S = '<span class="c-v">P₁</span>', P2S = '<span class="c-w">P₂</span>';
  const goldS = t => '<span class="c-gold">' + t + '</span>';

  /* ---------- numerics helpers ---------- */
  function sqrtSym(S) {
    const e = N.eigSym(S); const Q = N.fromCols(e.vectors);
    return N.matMul(N.matMul(Q, N.diag(e.values.map(l => Math.sqrt(Math.max(l, 0))))), N.transpose(Q));
  }
  const covN = X => N.cov(X, { ddof: 0 });
  function unitNoise(seed, n, d) {
    const Z = LA.rng.gaussianCloud(seed, n, { mean: new Array(d).fill(0), cov: N.identity(d) });
    const Zc = N.center(Z).Xc; const W = N.inv(sqrtSym(covN(Zc)));
    return Zc.map(z => N.matVec(W, z).map(x => Math.round(x * 1e4) / 1e4));
  }
  const chol2 = (sx, sy, rho) => [[sx, 0], [rho * sy, sy * Math.sqrt(Math.max(0, 1 - rho * rho))]];

  /** The cloud's Σ (exact: the noise is whitened) and its eigen-frame with q₂ = q₁ turned −90°. */
  function frame(s) {
    const S = s.src === 'axis' ? [[4, 0], [0, 0.25]] : [[1, s.rho], [s.rho, 1]];
    let q1, l1, l2;
    if (s.src === 'axis') { q1 = [1, 0]; l1 = 4; l2 = 0.25; }
    else if (Math.abs(s.rho) < 1e-6) { q1 = [R2, R2]; l1 = 1; l2 = 1; }
    else { q1 = s.rho > 0 ? [R2, R2] : [R2, -R2]; l1 = 1 + Math.abs(s.rho); l2 = 1 - Math.abs(s.rho); }
    const q2 = [q1[1], -q1[0]];
    const Q = N.fromCols([q1, q2]);
    const H = N.matMul(N.matMul(Q, N.diag([Math.sqrt(l1), Math.sqrt(l2)])), N.transpose(Q));
    return { S, Si: N.inv(S), q1, q2, l: [l1, l2], Q, H, th: Math.atan2(q1[1], q1[0]) };
  }
  /** Whitening in three moves, w ∈ [0,3]: rotate by Qᵀ, divide by √λ, rotate back by Q. */
  function whitenM(f, w) {
    const u1 = LA.clamp(w, 0, 1), u2 = LA.clamp(w - 1, 0, 1), u3 = LA.clamp(w - 2, 0, 1);
    const D = N.diag([LA.lerp(1, 1 / Math.sqrt(f.l[0]), u2), LA.lerp(1, 1 / Math.sqrt(f.l[1]), u2)]);
    return N.matMul(N.rot2(f.th * u3), N.matMul(D, N.rot2(-f.th * u1)));
  }
  const dM = (f, x) => Math.sqrt(Math.max(0, N.dot(x, N.matVec(f.Si, x))));
  function basePts(s) {
    if (s.src === 'axis') return s.z.map(p => [2 * p[0], 0.5 * p[1]]);
    const L = chol2(1, 1, s.rho); return s.z.map(p => N.matVec(L, p));
  }
  function viewM(s, f) {
    if (s.src === 'axis') return lerpM(I2(), s.Wd, 1);
    const sd = [1, 1];                         // both features have spread exactly 1 here
    const Dstd = N.diag([LA.lerp(1, 1 / sd[0], s.stdT), LA.lerp(1, 1 / sd[1], s.stdT)]);
    return N.matMul(Dstd, whitenM(f, s.whitenT));
  }

  LA.registerLesson({
    id: 'p4-11-whitening-mahalanobis',
    part: 4,
    title: 'Whitening, Mahalanobis distance, anomaly detection',
    subtitle: 'Two points at the same distance can be very differently unusual: measure in the cloud’s own units, and the ellipse becomes a circle.',
    minutes: 11,
    prereqs: ['p4-09-covariance', 'p4-10-pca', 'p1-05-norms-distance'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D', realSize: 'anomaly detection on 50 sensor channels uses a 50×50 Σ; LDA and Gaussian classifiers do this in d = 1000', transfers: 'd_M² = (x−μ)ᵀΣ⁻¹(x−μ) and W = Σ^−1/2 are identical in any dimension; the ellipse becomes an ellipsoid we cannot draw' },
    cantShow: ['density surfaces in d > 2 (the bell is implied by the contour rings)', 'why the χ² rule picks a particular threshold (statistics, not geometry)'],
    terms: [
      { term: 'Mahalanobis distance', def: 'Distance from the center measured in units of the cloud’s spread in that direction: d_M² = (x−μ)ᵀΣ⁻¹(x−μ).' },
      { term: 'whitening', def: 'A linear map W with WΣWᵀ = I that turns the cloud into a round unit ball.' },
      { term: 'whitening matrix', def: 'W = Σ^−1/2 = QΛ^−1/2Qᵀ: rotate into the gold axes, divide each axis by √λ, rotate back.' },
      { term: 'standardization', def: 'Dividing each feature by its own spread; it fixes the diagonal of Σ but leaves the tilt.' },
      { term: 'anomaly', def: 'A point whose Mahalanobis distance exceeds a chosen threshold.' },
    ],
    atlas: ['gaussian', 'normalization', 'optimization'],
    thumbnail: 4,

    state() {
      return {
        view: '2d', cam: cam2(),
        z: unitNoise(51, 200, 2), rho: 0.85, src: 'main',
        p1: [1.2 * R2, 1.2 * R2], p2: [1.2 * R2, -1.2 * R2],
        whitenT: 0, stdT: 0, rings: 0, ringAlpha: 1, threshold: 2.5, flag: false, showDensity: false, densT: 0,
        scene: 'main', showProbes: 1, showCircle: true, dims: false, ptsAlpha: 1, ringLabels: false, axesT: 0, dropT: 0,
        tileKind: null, cross: false, eqRings: false, probeHandles: 'none', threshHandle: false, whitenTrack: false,
        Wd: [[1, 0], [0, 1]], mlT: 0, panel: 'euclid',
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      renderMain(g, s, ctx);
    },

    math(s, tex) {
      const f = frame(s), P = [];
      const tn = (x, c) => tex.text(LA.fmt(x), { color: c });
      const p1 = tex.sym('P', { sub: '1', color: 'v' }), p2 = tex.sym('P', { sub: '2', color: 'w' });
      const dm = tex.sym('d', { sub: 'M' }), Sig = tex.sym('Σ');
      const quad = tex.T(tex.paren(tex.sym('x') + tex.op('−') + tex.sym('μ'))) + tex.inv(Sig) + tex.paren(tex.sym('x') + tex.op('−') + tex.sym('μ'));
      switch (s.panel) {
        case 'euclid':
          P.push(tex.eq(tex.norm(p1 + tex.op('−') + tex.sym('μ')), tex.op('='), tn(N.norm(s.p1)), tex.op(','), tex.norm(p2 + tex.op('−') + tex.sym('μ')), tex.op('='), tn(N.norm(s.p2))));
          break;
        case 'rings':
          P.push(tex.eq(tex.text('rings:'), quad, tex.op('='), tex.text('1, 4, 9', { color: 'teal' })));
          P.push(tex.eq(p1, tex.op(':'), tn(dM(f, s.p1), 'v'), tex.text('σ'), tex.op(','), p2, tex.op(':'), tn(dM(f, s.p2), 'w'), tex.text('σ')));
          break;
        case 'mahal':
          P.push(tex.eq(dm + '<sup>2</sup>', tex.op('='), tex.sum(tex.frac(tex.paren(tex.sym('x') + tex.op('·') + tex.sym('q', { sub: 'i', color: 'gold' })) + '<sup>2</sup>', tex.sym('λ', { sub: 'i' })), 'i', ''), tex.op('='), quad));
          P.push(tex.eq(dm + tex.paren(p1), tex.op('='), tn(dM(f, s.p1), 'v'), tex.op(','), dm + tex.paren(p2), tex.op('='), tn(dM(f, s.p2), 'w')));
          break;
        case 'rot':
          P.push(tex.eq(tex.sym('x′'), tex.op('='), tex.T(tex.sym('Q')), tex.paren(tex.sym('x') + tex.op('−') + tex.sym('μ'))));
          break;
        case 'scale': {
          const T = whitenM(f, s.whitenT);
          P.push(tex.eq(tex.sym('x″'), tex.op('='), tex.sym('Λ') + '<sup>−1/2</sup>', tex.sym('x′'), tex.op(','), tex.norm(tex.sym('x″')), tex.op('='), dm + tex.paren(tex.sym('x'))));
          P.push(tex.eq(tex.norm(p1), tex.op('='), tn(N.norm(N.matVec(T, s.p1)), 'v'), tex.op(','), tex.norm(p2), tex.op('='), tn(N.norm(N.matVec(T, s.p2)), 'w')));
          break; }
        case 'W': {
          const W = whitenM(f, 3), I = N.matMul(N.matMul(W, f.S), N.transpose(W));
          P.push(tex.eq(tex.sym('W'), tex.op('='), tex.sym('Q'), tex.sym('Λ') + '<sup>−1/2</sup>', tex.T(tex.sym('Q')), tex.op('='), tex.mat(W, { colors: 'gold' })));
          P.push(tex.eq(tex.sym('W'), Sig, tex.T(tex.sym('W')), tex.op('='), tex.mat(I.map(r => r.map(x => Math.abs(x) < 5e-3 ? 0 : x)), { colors: 'ok' }), tex.op('='), tex.sym('I')));
          break; }
        case 'std':
          P.push(tex.eq(tex.sym('D') + '<sup>−1/2</sup>', Sig, tex.sym('D') + '<sup>−1/2</sup>', tex.op('='), tex.mat(f.S, { colors: (i, j) => i === j ? 'text' : 'bad' }), tex.op('≠'), tex.sym('I')));
          break;
        case 'anom': {
          const n = basePts(s).filter(p => dM(f, p) > s.threshold).length;
          P.push(tex.eq(tex.text('anomaly'), tex.op('⇔'), dm + tex.paren(tex.sym('x')), tex.op('>'), tex.sym('τ'), tex.op(','), tex.sym('τ'), tex.op('='), tn(s.threshold, 'gold')));
          P.push(tex.eq(tex.text('flagged'), tex.op('='), tn(n, 'bad'), tex.text('of 200'), tex.op(';'), dm + tex.paren(p2), tex.op('='), tn(dM(f, s.p2), 'w')));
          break; }
        case 'dens':
          P.push(tex.eq(tex.sym('p') + tex.paren(tex.sym('x')), tex.op('∝'), tex.text('exp'), tex.paren(tex.op('−') + dm + '<sup>2</sup>' + tex.op('/') + tex.text('2')), tex.op('/'), tex.sqrt(tex.text('det ') + Sig)));
          P.push(tex.eq(tex.text('det'), Sig, tex.op('='), tn(N.det(f.S))));
          break;
        case 'axis':
          P.push(tex.eq(tex.sym('λ'), tex.op('='), tex.text('(4, 0.25)'), tex.op(';'), tex.text('your'), tex.sym('Λ') + '<sup>−1/2</sup>', tex.op('='), tex.mat(s.Wd, { colors: 'gold' })));
          break;
        case 'ml': case 'recap':
          if (s.panel === 'recap') P.push(tex.eq(tex.sym('W'), tex.op('='), Sig + '<sup>−1/2</sup>'));
          P.push(tex.eq(dm + '<sup>2</sup>', tex.op('='), quad));
          break;
        default: { // play
          const W = whitenM(f, 3);
          P.push(tex.eq(dm + tex.paren(p1), tex.op('='), tn(dM(f, s.p1), 'v'), tex.op(','), dm + tex.paren(p2), tex.op('='), tn(dM(f, s.p2), 'w')));
          P.push(tex.eq(tex.text('Euclid:'), tn(N.norm(s.p1), 'v'), tex.op(','), tn(N.norm(s.p2), 'w'), tex.op(';'), tex.text('flagged'), tex.op('='), tn(basePts(s).filter(p => dM(f, p) > s.threshold).length, 'bad')));
          P.push(tex.eq(tex.sym('W'), tex.op('='), Sig + '<sup>−1/2</sup>', tex.op('='), tex.mat(W, { colors: 'gold' })));
        }
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Two probes, same distance', duration: 3.5,
        narration: 'A tilted cloud of 200 synthetic points and two new observations, the yellow ' + P1S + ' and the purple ' + P2S + '. Both are exactly 1.2 units from the center. Yet one sits comfortably among the data and the other hangs in empty space. Plain distance cannot tell them apart.',
        setup(s) { reset(s, { panel: 'euclid', rings: 1, showProbes: 0, dims: false }); },
        at(s, t) { s.showProbes = seg(t, 0.15, 0.45); s.dims = t > 0.55; } },

      { title: 'Rings of equal “unusualness”', duration: 3.5,
        narration: 'The cloud’s own rulers are the ellipses of Σ: the teal 1σ, 2σ and 3σ rings. Points on the same ring are equally typical. ' + P1S + ' lies inside the 1σ ring; ' + P2S + ' sits beyond 3σ. The right distance counts rings, not units.',
        setup(s) { reset(s, { panel: 'rings', rings: 1 }); },
        at(s, t) { s.rings = 1 + 2 * seg(t, 0.05, 0.6); s.ringLabels = t > 0.65; } },

      { title: 'The Mahalanobis distance', duration: 4.5,
        narration: 'Measure along each gold axis in units of that axis’s spread: along ' + goldS('q₁') + ' divide by √λ₁ = 1.36, along ' + goldS('q₂') + ' by √λ₂ = 0.39. Then use Pythagoras. That is the <b>Mahalanobis distance</b>, d<sub>M</sub>² = (x−μ)ᵀΣ⁻¹(x−μ). P₁: 1.2 / 1.36 = 0.88. P₂: 1.2 / 0.39 = 3.1.',
        setup(s) { reset(s, { panel: 'mahal', rings: 3, ringAlpha: 0.45, ringLabels: false }); },
        at(s, t) { s.axesT = seg(t, 0, 0.3); s.dropT = seg(t, 0.35, 0.9); } },

      { title: 'Whitening, move 1: rotate into the gold frame', duration: 3,
        narration: 'Here is the same idea as a transformation of the whole picture: <b>whitening</b>. First rotate everything by Qᵀ so the gold axes become the grid axes. The cloud is now axis-aligned but still elongated; the rings and both probes ride along.',
        setup(s) { reset(s, { panel: 'rot', rings: 3, axesT: 1, showCircle: true }); },
        at(s, t) { s.whitenT = seg(t, 0.1, 0.9); } },

      { title: 'Move 2: divide each axis by its spread', duration: 3.5,
        narration: 'Now squeeze the long axis by 1/√λ₁ and stretch the short one by 1/√λ₂. The ellipses become circles of radius 1, 2, 3 and the cloud becomes a round ball. In this picture ordinary Euclidean distance <i>is</i> Mahalanobis distance: P₁ now sits 0.88 from the center, P₂ 3.1.',
        setup(s) { reset(s, { panel: 'scale', rings: 3, axesT: 1, whitenT: 1 }); },
        at(s, t) { s.whitenT = 1 + seg(t, 0.05, 0.75); s.eqRings = t > 0.85; } },

      { title: 'Move 3 (optional): rotate back — the whitening matrix W', duration: 3.5,
        narration: 'Rotating back by Q keeps the cloud round and restores the features’ original orientation. The whole map is the <b>whitening matrix</b> W = QΛ<sup>−1/2</sup>Qᵀ = Σ<sup>−1/2</sup>: a symmetric matrix with WΣWᵀ = I. Any W with that property whitens; this one is the “least rotating” choice.',
        setup(s) { reset(s, { panel: 'W', rings: 3, axesT: 1, whitenT: 2 }); },
        at(s, t) { s.whitenT = 2 + seg(t, 0.05, 0.7); s.tileKind = t > 0.6 ? 'W' : null; } },

      { title: 'Misconception: standardizing is whitening', duration: 4, misconception: true,
        narration: '✕ “Divide each feature by its own standard deviation and you are done.” Both features here already have spread 1, so this <b>standardization</b> changes <i>nothing</i>: the cloud stays tilted and P₂ still looks close. Whitening needs the rotation — it uses the off-diagonal of Σ, which standardizing ignores.',
        setup(s) { reset(s, { panel: 'std', rings: 1, whitenT: 0, tileKind: 'S' }); },
        at(s, t) { s.stdT = seg(t, 0.1, 0.5); s.cross = t > 0.55; } },

      { title: 'Anomaly detection: a threshold on the distance', duration: 5, interactive: true,
        narration: 'Set a threshold: anything beyond 2.5σ is an <b>anomaly</b>. Drag the gold threshold ring and watch points flip red as it shrinks; drag P₂ along the dashed circle of radius 1.2 and see it turn from normal to anomalous as it moves from the long axis to the short one. Distance in the data’s own units.',
        setup(s) { reset(s, { panel: 'anom', rings: 0, flag: true, threshHandle: true, probeHandles: 'circle', showProbes: 1 }); },
        at(s, t) {
          s.threshold = LA.lerp(3.4, 2.5, seg(t, 0.05, 0.4));
          const a = LA.lerp(Math.PI / 4, -Math.PI / 4, seg(t, 0.45, 0.95));
          s.p2 = [1.2 * Math.cos(a), 1.2 * Math.sin(a)];
        } },

      { title: 'The density as a bell', duration: 3.5,
        narration: 'The Gaussian’s density depends on x only through d<sub>M</sub>: it falls like e<sup>−d²/2</sup>, and the rings are its level curves. The determinant det Σ from p4-09 scales its height so the total is 1. Low density, large d<sub>M</sub>, anomaly: three names, one picture.',
        setup(s) { reset(s, { panel: 'dens', rings: 3, ringAlpha: 0.9, densT: 0, ptsAlpha: 0.35, showCircle: false }); },
        at(s, t) { s.densT = seg(t, 0.05, 0.6); } },

      { title: 'Why ML cares', duration: 4, mlTie: true,
        narration: 'Fraud and fault detection flag large Mahalanobis distances. Linear discriminant analysis classifies by it. Whitening decorrelates inputs so gradient descent sees a round bowl instead of a valley (p5-03), and some normalization layers approximate it. The picture stops at 3. The formula does not: the mechanism is identical in d = 50.',
        setup(s) { reset(s, { panel: 'ml', scene: 'ml', mlT: 0 }); },
        at(s, t) { s.mlT = seg(t, 0.1, 0.85); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Equal Euclidean distance is not equal rarity. Mahalanobis divides by the spread along each gold axis — it is Euclidean distance after whitening. Whitening: rotate, rescale by 1/√λ, (rotate back); W = Σ<sup>−1/2</sup>. Standardizing alone leaves the tilt. Threshold d<sub>M</sub> to flag anomalies.',
        cards: ['<b>Rings, not circles.</b> P₁ and P₂ are both 1.2 away, but at 0.88σ and 3.1σ.', '<b>Rotate, rescale, (rotate back) = Σ<sup>−1/2</sup>.</b> The ellipse becomes a circle; Euclid becomes Mahalanobis.', '<b>Threshold d<sub>M</sub>.</b> Points outside the gold ring d<sub>M</sub> = τ are flagged as anomalies.'],
        setup(s) { reset(s, { panel: 'recap', scene: 'recap' }); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the yellow <span class="c-v">P₁</span> and purple <span class="c-w">P₂</span> anywhere, change the correlation, run the whitening slider through its three moves, and move the threshold τ. Both distances, the flagged count and W update live.</p>',
      setup(s) { reset(s, { panel: 'play', rings: 3, ringAlpha: 0.7, axesT: 1, flag: true, probeHandles: 'free', threshHandle: true }); },
      controls: [
        { type: 'slider', bind: 'rho', min: -0.95, max: 0.95, step: 0.01, label: 'correlation ρ', color: 'b' },
        { type: 'slider', bind: 'whitenT', min: 0, max: 3, step: 0.01, label: 'whitening: rotate / scale / back', color: 'gold', fmt: x => x < 0.01 ? 'original' : x < 1 ? 'rotating…' : x < 1.01 ? 'rotated' : x < 2 ? 'scaling…' : x < 2.01 ? 'round, axis-aligned' : x < 3 ? 'rotating back…' : 'W = Σ^−1/2' },
        { type: 'slider', bind: 'threshold', min: 1, max: 4, step: 0.05, label: 'threshold τ', color: 'gold' },
        { type: 'preset', label: 'View', items: [
          { label: '2D picture', apply(s) { s.showDensity = false; s.view = '2d'; s.cam = cam2(); } },
          { label: 'Density bell (3D, drag to orbit)', apply(s) { s.showDensity = true; s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), { dist: 11, yaw: 0.9, pitch: 0.55 }); } },
        ] },
        { type: 'preset', label: 'Presets', items: [
          { label: 'ρ = 0.85', values: { rho: 0.85, whitenT: 0 } },
          { label: 'Round (ρ = 0)', values: { rho: 0, whitenT: 0 } },
          { label: 'Nearly flat (ρ = 0.98)', values: { rho: 0.98, whitenT: 0 } },
          { label: 'Whitened', values: { whitenT: 3 } },
        ] },
        { type: 'info', html: '<b>Tip.</b> Put P₁ and P₂ on the same dashed circle: equal Euclidean distance, very different d<sub>M</sub>.' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag ' + P2S + ' to Mahalanobis distance 1 (onto the inner teal 1σ ring).',
        setup(s) { reset(s, { panel: 'rings', rings: 3, ringLabels: false, probeHandles: 'p2' }); s.p2 = [1.2 * R2, -1.2 * R2]; },
        judge(s) { const d = dM(frame(s), s.p2); return { ok: Math.abs(d - 1) < LA.TOL.scalar, progress: Math.max(0, 1 - Math.abs(d - 1) / 2.1) }; },
        hint: 'Move toward the center until it touches the inner teal ring.',
        solution(s) { const f = frame(s); s.p2 = N.scale(f.q2, Math.sqrt(f.l[1])); } },
      { kind: 'manipulate',
        prompt: 'Drag the white handle on the whitening track to the stage where the cloud is round <i>and</i> axis-aligned.',
        setup(s) { reset(s, { panel: 'scale', rings: 3, axesT: 1, whitenT: 0, whitenTrack: true }); },
        judge(s) { const d = Math.abs(s.whitenT - 2); return { ok: d < 0.15, progress: Math.max(0, 1 - d / 2) }; },
        hint: 'After the rotation and the rescale, before rotating back.',
        solution(s) { s.whitenT = 2; } },
      { kind: 'entry',
        prompt: 'This cloud has axis variances λ = (4, 0.25). Type the whitening matrix Λ<sup>−1/2</sup>; the cloud is multiplied by what you type.',
        bind: 'Wd', answer: [[0.5, 0], [0, 2]], tol: LA.TOL.entry, label: 'Λ^−1/2', colors: 'gold',
        setup(s) { reset(s, { panel: 'axis', src: 'axis', rings: 1, showProbes: 0, showCircle: false }); s.Wd = [[1, 0], [0, 1]]; } },
    ],
  });

  /* ---------- helpers ---------- */
  function cam2() { return Object.assign(LA.cam.defaults2d(), { center: [0, 0], halfH: 4.2 }); }
  function reset(s, o) {
    Object.assign(s, {
      scene: 'main', view: '2d', src: 'main', rho: 0.85, whitenT: 0, stdT: 0, rings: 0, ringAlpha: 1, flag: false, densT: 0,
      showProbes: 1, showCircle: true, dims: false, ptsAlpha: 1, ringLabels: false, axesT: 0, dropT: 0, tileKind: null, cross: false,
      eqRings: false, probeHandles: 'none', threshHandle: false, whitenTrack: false, threshold: 2.5, mlT: 0, showDensity: false,
      p1: [1.2 * R2, 1.2 * R2], p2: [1.2 * R2, -1.2 * R2],
    }, o || {});
    s.cam = cam2();
  }

  function tile2(g, M, o) {
    const x0 = g.size.w - 230, y0 = o.y || 30;
    g.label(o.name, [x0, y0 + 13], { screen: true, anchor: 'left', size: 17, color: 'hud', font: 'math' });
    g.label('[', [x0 + 46, y0 + 12], { screen: true, size: 48, color: 'muted', font: 'ui', bg: false });
    g.label(']', [x0 + 176, y0 + 12], { screen: true, size: 48, color: 'muted', font: 'ui', bg: false });
    const cx = [x0 + 84, x0 + 142], cy = [y0, y0 + 26];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) g.label(LA.fmt.fixed(M[i][j], 2), [cx[j], cy[i]], { screen: true, size: 15.5, color: o.colorAt(i, j), font: 'mono', bg: false });
    return (i, j) => [cx[j], cy[i]];
  }

  /* ---------- main scene ---------- */
  function renderMain(g, s, ctx) {
    if (s.showDensity && s.view === '3d') { renderBell(g, s); return; }
    const f = frame(s), T = viewM(s, f), Ti = N.inv(T);
    const B = basePts(s), P = B.map(p => N.matVec(T, p));
    g.ghostGrid();
    if (s.whitenT > 0.001 || s.src === 'axis') g.grid(T, { alpha: 0.45, range: 8 });
    else g.axes();
    // density underlay
    if (s.densT > 0) {
      const fn = x => Math.exp(-0.5 * dM(f, N.matVec(Ti, x)) ** 2);
      g.heat2d(fn, { x0: -4.6, x1: 4.6, y0: -3.8, y1: 3.8 }, { res: 64, range: [0, 1], alpha: 0.75 * s.densT });
      g.contours(fn, [0.5, 1, 1.5, 2, 3].map(d => Math.exp(-d * d / 2)), { x0: -4.6, x1: 4.6, y0: -3.8, y1: 3.8 }, { res: 60, colors: ['muted', 'teal', 'muted', 'teal', 'teal'], width: 1.3, alpha: s.densT });
    }
    // rings 1σ…3σ
    for (let k = 1; k <= 3; k++) {
      const a = LA.clamp(s.rings - k + 1, 0, 1); if (a <= 0) continue;
      const M = N.scaleM(N.matMul(T, f.H), k);
      g.ellipse(M, [0, 0], { stroke: 'teal', width: k === 1 ? 2.2 : 1.6, strokeAlpha: a * s.ringAlpha * (1.1 - 0.25 * k) });
      if (s.ringLabels || s.whitenT > 1.5) g.label(k + 'σ', N.matVec(M, N.add(N.scale(f.q1, -0.55), N.scale(f.q2, -0.835))), { color: 'teal', size: 12, font: 'ui', alpha: a, dx: -10, dy: 8 });
    }
    // the threshold ring
    if (s.flag) {
      const M = N.scaleM(N.matMul(T, f.H), s.threshold);
      g.ellipse(M, [0, 0], { stroke: 'gold', width: 2, dashed: true });
      const tip = N.matVec(N.matMul(T, N.scaleM(f.H, s.threshold)), f.q1);
      g.label('τ = ' + LA.fmt.fixed(s.threshold, 2) + 'σ', tip, { color: 'gold', size: 13, font: 'ui', dx: 30, dy: -10 });
      if (s.threshHandle) g.handle('tau', tip, (p, st) => { const ff = frame(st), TT = viewM(st, ff); const x = N.matVec(N.inv(TT), [p[0], p[1]]); st.threshold = LA.clamp(dM(ff, x), 0.5, 4.5); }, { color: 'white' });
    }
    // the dashed circle of radius 1.2 (in the original picture)
    if (s.showCircle && s.showProbes > 0.01) g.ellipse(N.scaleM(T, 1.2), [0, 0], { stroke: 'hud', width: 1.2, dashed: true, strokeAlpha: 0.6 * s.showProbes });
    // points
    const nFlag = { n: 0 };
    g.points(P, { r: 3, alpha: 0.85 * s.ptsAlpha, colors: i => { if (s.flag && dM(f, B[i]) > s.threshold) { nFlag.n++; return 'bad'; } return 'hud'; } });
    // gold axes with √λ unit ticks
    if (s.axesT > 0) {
      [f.q1, f.q2].forEach((q, i) => {
        const L = Math.sqrt(f.l[i]);
        const tip = N.matVec(T, N.scale(q, 3 * L * s.axesT)), neg = N.matVec(T, N.scale(q, -3 * L * s.axesT));
        g.line(neg, tip, { color: 'gold', width: 1.4, alpha: 0.55 });
        g.vector(N.matVec(T, N.scale(q, L * s.axesT)), { color: 'gold', width: 3, head: 10, label: i ? 'q₂' : 'q₁', labelFont: 'math' });
        const nrm = N.matVec(T, [-q[1], q[0]]), nn = N.normalize(nrm);
        for (let k = -3; k <= 3; k++) { if (!k) continue; const c = N.matVec(T, N.scale(q, k * L)); g.line(N.add(c, N.scale(nn, 0.07)), N.sub(c, N.scale(nn, 0.07)), { color: 'gold', width: 1.4 }); }
      });
      if (s.panel === 'mahal' && s.axesT > 0.9) g.hud('gold ticks: along q₁ every √λ₁ = ' + LA.fmt.fixed(Math.sqrt(f.l[0]), 2) + ', along q₂ every √λ₂ = ' + LA.fmt.fixed(Math.sqrt(f.l[1]), 2), 16, g.size.h - 22, { size: 13.5, color: 'gold', bold: true });
    }
    // the probes
    if (s.showProbes > 0.01) {
      const tp1 = N.matVec(T, s.p1), tp2 = N.matVec(T, s.p2);
      const d1 = dM(f, s.p1), d2 = dM(f, s.p2);
      if (s.dims) { g.dim([0, 0], tp1, { label: LA.fmt.fixed(N.norm(s.p1), 1), color: 'v', offset: -14 }); g.dim([0, 0], tp2, { label: LA.fmt.fixed(N.norm(s.p2), 1), color: 'w', offset: 14 }); }
      if (s.dropT > 0) drawDrops(g, s, f, T);
      const bad2 = s.flag ? d2 > s.threshold : (s.ringLabels && d2 > 3);
      g.point(tp1, { r: 7 * s.showProbes, color: 'v', label: 'P₁', stroke: 'white', strokeWidth: 1 });
      g.point(tp2, { r: 7 * s.showProbes, color: 'w', label: 'P₂', stroke: bad2 ? 'bad' : 'white', strokeWidth: bad2 ? 3 : 1 });
      if (s.ringLabels || s.panel === 'anom' || s.panel === 'play') {
        g.label(LA.fmt.fixed(d1, 2) + 'σ', tp1, { color: 'v', size: 13, font: 'ui', dx: 30, dy: 10 });
        g.label(LA.fmt.fixed(d2, 2) + 'σ', tp2, { color: bad2 ? 'bad' : 'w', size: 13, font: 'ui', dx: 30, dy: 12 });
      }
      if (s.whitenT > 1.02 && s.panel !== 'play') {
        g.label('|P₁| = ' + LA.fmt.fixed(N.norm(tp1), 2), tp1, { color: 'v', size: 13, font: 'ui', dx: 0, dy: -24 });
        g.label('|P₂| = ' + LA.fmt.fixed(N.norm(tp2), 2), tp2, { color: 'w', size: 13, font: 'ui', dx: 22, dy: 0, anchor: 'left' });
      }
      if (s.eqRings) { g.marker.ring(tp1, { r: 13, color: 'gold' }); g.marker.ring(tp2, { r: 13, color: 'gold' }); g.hud('Euclidean length now = Mahalanobis distance', 16, g.size.h - 22, { size: 14, color: 'gold', bold: true }); }
      // handles
      const onP = (key, circle) => (p, st) => { const ff = frame(st), TT = viewM(st, ff); let x = N.matVec(N.inv(TT), [p[0], p[1]]); if (circle) x = N.scale(N.normalize(x), 1.2); st[key] = x; };
      if (s.probeHandles === 'free') { g.handle('p1', tp1, onP('p1', false), { color: 'v' }); g.handle('p2', tp2, onP('p2', false), { color: 'w' }); }
      if (s.probeHandles === 'p2') g.handle('p2', tp2, onP('p2', false), { color: 'w' });
      if (s.probeHandles === 'circle') g.handle('p2', tp2, onP('p2', true), { color: 'w' });
    }
    if (s.flag) g.hud('flagged: ' + nFlag.n + ' of ' + B.length, 16, 24, { size: 14, color: 'bad', bold: true });
    // HUD bars of d_M (step 3)
    if (s.dropT > 0.95) {
      const d1 = dM(f, s.p1), d2 = dM(f, s.p2);
      g.bars([d1, d2], { screen: true, x: 22, y: 70, w: 90, h: 120, max: 3.5, colors: ['v', 'w'], labels: ['P₁', 'P₂'], values: true, title: 'd_M (in σ)' });
    }
    // tiles
    if (s.tileKind === 'W') {
      const W = whitenM(f, 3), I = N.matMul(N.matMul(W, f.S), N.transpose(W));
      tile2(g, W, { name: 'W =', colorAt: () => 'gold' });
      const ok = Math.abs(I[0][0] - 1) + Math.abs(I[1][1] - 1) + Math.abs(I[0][1]) < 1e-6;
      g.hud('WΣWᵀ = I', g.size.w - 200, 98, { size: 14, color: ok ? 'ok' : 'bad', bold: true });
      if (ok) g.marker.check([g.size.w - 100, 98], { screen: true, size: 18 });
    }
    if (s.tileKind === 'S') {
      const sd = [1, 1];
      const Sstd = [[f.S[0][0] / (sd[0] * sd[0]), f.S[0][1] / (sd[0] * sd[1])], [f.S[1][0] / (sd[0] * sd[1]), f.S[1][1] / (sd[1] * sd[1])]];
      const at = tile2(g, Sstd, { name: 'Σ_std =', colorAt: (i, j) => i === j ? 'text' : (s.cross ? 'bad' : 'b') });
      if (s.cross) {
        const c = at(0, 1); g.marker.cross([c[0] + 40, c[1]], { screen: true, size: 22 });
        g.label('still tilted, P₂ still “near”', [g.size.w - 130, 100], { screen: true, color: 'bad', size: 13, font: 'ui' });
        g.marker.ring(N.matVec(T, s.p2), { r: 14, color: 'bad' });
      }
      g.hud(s.stdT > 0.5 ? 'standardized: diag(1/σx, 1/σy) = I — nothing moved' : 'standardizing…', 16, g.size.h - 22, { size: 14, color: 'hud', bold: true });
    }
    if (s.whitenTrack) drawTrack(g, s);
    if (s.src === 'axis') g.hud('synthetic cloud with axis variances 4 and 0.25 — round means you typed Λ^−1/2', 16, 24, { size: 13, color: 'hud' });
    if (s.densT > 0) g.hud('density: brighter = more likely; teal contours at d_M = 1, 2, 3', 16, 24, { size: 13, color: 'hud' });
  }

  function drawDrops(g, s, f, T) {
    const u = s.dropT;
    [[s.p1, 'v', 1], [s.p2, 'w', -1]].forEach(([p, col]) => {
      [f.q1, f.q2].forEach((q, i) => {
        const c = N.dot(p, q); if (Math.abs(c) < 1e-6) return;
        const foot = N.matVec(T, N.scale(q, c * u)), tp = N.matVec(T, p);
        g.line([0, 0], foot, { color: col, width: 4, alpha: 0.6 });
        if (u > 0.95) g.line(tp, N.matVec(T, N.scale(q, c)), { color: 'residual', dashed: [3, 3], width: 1.2 });
        if (u > 0.6) {
          const L = Math.sqrt(f.l[i]);
          g.label(LA.fmt.fixed(Math.abs(c), 2) + ' ÷ ' + LA.fmt.fixed(L, 2) + ' = ' + LA.fmt.fixed(Math.abs(c) / L, 2), N.matVec(T, N.scale(q, c * 0.5)), { color: col, size: 12.5, font: 'ui', dx: i ? 80 : -60, dy: i ? -14 : -20, alpha: seg(u, 0.6, 0.9) });
        }
      });
    });
  }

  function drawTrack(g, s) {
    const y = -3.3, x0 = -2.4, x1 = 2.4, X = w => x0 + (x1 - x0) * w / 3;
    g.line([x0, y], [x1, y], { color: 'muted', width: 2, alpha: 0.8 });
    [0, 1, 2, 3].forEach(k => { g.line([X(k), y - 0.08], [X(k), y + 0.08], { color: 'muted', width: 1.5 }); });
    ['rotate', 'scale', 'rotate back'].forEach((t, k) => g.label(t, [X(k + 0.5), y], { anchor: 'below', dy: 8, color: 'muted', size: 11.5, font: 'ui' }));
    const hp = [X(s.whitenT), y];
    g.point(hp, { r: 4, color: 'gold' });
    g.handle('wt', hp, (p, st) => { st.whitenT = LA.clamp((p[0] - x0) / (x1 - x0) * 3, 0, 3); }, { color: 'white' });
  }

  /* ---------- 3D bell (playground View preset; s.view = '3d', so the user can orbit) ---------- */
  function renderBell(g, s) {
    const f = frame(s), h = 2.2, B = { x0: -3, x1: 3, y0: -3, y1: 3 };
    const fn = x => h * Math.exp(-0.5 * dM(f, x) ** 2);
    g.grid(null, { range: 3, alpha: 0.3 });
    g.surface(fn, B, { res: 24, mode: 'both', color: 'teal', alpha: 0.5, zRange: [0, h] });
    g.contours(fn, [1, 2, 3].map(d => h * Math.exp(-d * d / 2)), B, { res: 40, color: 'teal', width: 1.6, z: 0 });
    g.contours(fn, [h * Math.exp(-s.threshold * s.threshold / 2)], B, { res: 40, color: 'gold', width: 2, z: 0 });
    g.point([s.p1[0], s.p1[1], fn(s.p1)], { r: 6, color: 'v', label: 'P₁' });
    g.point([s.p2[0], s.p2[1], fn(s.p2)], { r: 6, color: 'w', label: 'P₂' });
    g.hud('density p(x) ∝ exp(−d_M²/2): the teal rings (1σ, 2σ, 3σ) are its level curves', 14, 24, { size: 13.5, color: 'hud' });
    g.hud('gold floor ring: the threshold τ = ' + LA.fmt.fixed(s.threshold, 2) + 'σ · drag to orbit · the whitening slider acts on the 2D picture', 14, 44, { size: 12.5, color: 'muted' });
  }

  /* ---------- ML tie-in rows ---------- */
  function renderML(g, s) {
    const u = s.mlT;
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 2.3 } }, (g2, r) => {
      const rand = LA.rng(52), vals = []; for (let i = 0; i < 50; i++) vals.push(0.8 * LA.rng.gauss(rand));
      vals[31] = LA.lerp(vals[31], 3.6, u);
      const x = r.x + 16, y = r.y + r.h * 0.48, w = r.w - 32;
      g2.strip(vals, { x, y, w, h: 26, range: [-4, 4], ramp: t => C.diverge(2 * t - 1), title: '50 sensor channels (synthetic), one reading', highlight: u > 0.6 ? 31 : undefined });
      if (u > 0.6) { g2.marker.cross([x + (31.5 / 50) * w, y + 46], { screen: true, size: 16 }); g2.label('d_M = 4.2 > τ → flagged', [x + (31.5 / 50) * w, y + 66], { screen: true, color: 'bad', size: 12.5, font: 'ui' }); }
      g2.hud('fault & fraud detection', r.x + 12, r.y + 22, { size: 13.5, color: 'bad', bold: true });
      g2.hud(HONEST, r.x + 12, r.y + r.h - 16, { size: 12.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [-1.4, 0], halfH: 2.3 } }, (g2, r) => {
      const S = [[1, 0.7], [0.7, 1]], H = sqrtSym(S), m0 = [-0.9, 0.5], m1 = [0.9, -0.5];
      const w = N.matVec(N.inv(S), N.sub(m1, m0)), mid = N.scale(N.add(m0, m1), 0.5);
      g2.ellipse(N.scaleM(H, 1.6), m0, { stroke: 'data', width: 2 }); g2.ellipse(N.scaleM(H, 1.6), m1, { stroke: 'dataB', width: 2 });
      g2.point(m0, { r: 4, color: 'data' }); g2.point(m1, { r: 4, color: 'dataB' });
      g2.lineThrough(mid, [-w[1], w[0]], { color: 'gold', width: 2, len: 3 * u + 0.01 });
      g2.hud('LDA: classify by Mahalanobis distance', r.x + 12, r.y + 22, { size: 13.5, color: 'gold', bold: true });
      g2.hud('the gold boundary: equal d_M to both classes', r.x + 12, r.y + 42, { size: 12, color: 'muted' });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [-1.4, 0], halfH: 2.3 } }, (g2, r) => {
      const A = lerpM([[2.2, 1.6], [1.6, 1.4]], I2(), u), H = sqrtSym(A);
      [0.5, 1, 1.5, 2].forEach(k => g2.ellipse(N.scaleM(H, k * 0.8), [0, 0], { stroke: 'teal', width: 1.4, strokeAlpha: 0.9 - 0.15 * k }));
      g2.point([0, 0], { r: 4, color: 'gold' });
      g2.hud('whitened inputs → a round loss bowl (p5-03)', r.x + 12, r.y + 22, { size: 13.5, color: 'teal', bold: true });
      g2.hud(u < 0.95 ? 'elongated contours become circles…' : 'gradient descent heads straight to the minimum', r.x + 12, r.y + 42, { size: 12, color: 'muted' });
    });
  }

  /* ---------- recap: three stacked mini-cards ---------- */
  function renderRecap(g, s) {
    const f = frame(s), B = basePts(s);
    const cam = { center: [-1.6, 0], halfH: 2.3 };
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam }, (g2, r) => {
      g2.points(B.map(p => N.scale(p, 0.6)), { r: 1.8, color: 'hud', alpha: 0.5 });
      [1, 2, 3].forEach(k => g2.ellipse(N.scaleM(f.H, k * 0.6), [0, 0], { stroke: 'teal', width: 1.4, strokeAlpha: 1 - 0.25 * k }));
      g2.circle([0, 0], 1.2 * 0.6, { stroke: 'hud', width: 1, dashed: true });
      g2.point(N.scale(s.p1, 0.6), { r: 5, color: 'v', label: 'P₁ 0.88σ' }); g2.point(N.scale(s.p2, 0.6), { r: 5, color: 'w', label: 'P₂ 3.1σ' });
      g2.hud('1 · Rings, not circles', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam }, (g2, r) => {
      g2.ellipse(N.scaleM(f.H, 0.9), [-0.9, 0], { stroke: 'teal', width: 2 });
      g2.vector([0.6, 0], [1.6, 0], { color: 'gold', width: 2.4, head: 10, label: 'W', labelFont: 'math' });
      g2.circle([2.8, 0], 0.9, { stroke: 'teal', width: 2 });
      g2.hud('2 · Rotate, rescale, (rotate back): W = Σ^−1/2', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam }, (g2, r) => {
      const sc = 0.55;
      g2.points(B.map(p => N.scale(p, sc)), { r: 2, colors: i => dM(f, B[i]) > 2.5 ? 'bad' : 'hud', alpha: 0.85 });
      g2.ellipse(N.scaleM(f.H, 2.5 * sc), [0, 0], { stroke: 'gold', width: 2, dashed: true });
      g2.hud('3 · Threshold d_M: outside the gold ring = anomaly', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
  }
})();
