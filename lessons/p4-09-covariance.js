/* p4-09 — Mean, centering, covariance as a shape; Gaussians (Σ = XcᵀXc/n, the ellipse of Σ, x = μ + Lz). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpM = LA.lerpM, lerpV = LA.lerpV, C = LA.color;
  const HONEST = 'The picture stops at 3. The formula does not.';
  const I2 = () => [[1, 0], [0, 1]];
  const MU = [2, 1];
  const SIG = [[2, 1.2], [1.2, 1]];
  const RHO0 = 1.2 / Math.sqrt(2);                       // 0.8485…: the correlation of SIG
  const SUB = ['₀', '₁', '₂', '₃'];
  const goldS = t => '<span class="c-gold">' + t + '</span>';

  /* ---------- numerics helpers (all deterministic) ---------- */
  function sqrtSym(S) {
    const e = N.eigSym(S); const Q = N.fromCols(e.vectors);
    return N.matMul(N.matMul(Q, N.diag(e.values.map(l => Math.sqrt(Math.max(l, 0))))), N.transpose(Q));
  }
  const covN = X => N.cov(X, { ddof: 0 });             // this lesson divides by n (the narration says /n)
  /** Unit noise whose sample mean is exactly 0 and whose /n covariance is exactly I (a whitened LA.rng cloud). */
  function unitNoise(seed, n, d) {
    const Z = LA.rng.gaussianCloud(seed, n, { mean: new Array(d).fill(0), cov: N.identity(d) });
    const Zc = N.center(Z).Xc; const W = N.inv(sqrtSym(covN(Zc)));
    return Zc.map(z => N.matVec(W, z).map(x => Math.round(x * 1e4) / 1e4));
  }
  const chol2 = (sx, sy, rho) => [[sx, 0], [rho * sy, sy * Math.sqrt(Math.max(0, 1 - rho * rho))]];
  const genSigma = (sx, sy, rho) => [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]];
  const L0 = chol2(Math.sqrt(2), 1, RHO0);              // = chol(SIG) = [[1.41,0],[0.85,0.53]]

  // the 3D pancake: L3 = R·diag(3, 1.2, 0.3), R = rot3(z, 35°)·rot3(x, 25°)
  const R3 = N.matMul(N.rot3([0, 0, 1], 35 * Math.PI / 180), N.rot3([1, 0, 0], 25 * Math.PI / 180));
  const L3 = N.matMul(R3, N.diag([3, 1.2, 0.3]));
  const S3 = N.matMul(L3, N.transpose(L3));
  const S3H = N.matMul(L3, N.transpose(R3));             // Σ₃^½ = R·diag(√λ)·Rᵀ
  const E3 = N.eigSym(S3);

  // 64-dimensional example: covariance of synthetic 8×8 digits (computed once, lazily; not stored in s)
  let DIG = null;
  function digits() {
    if (DIG) return DIG;
    const D = LA.rng.syntheticDigits(49, 90, { size: 8 });
    const S = covN(D.map(d => d.pixels));
    let m = 0; S.forEach(r => r.forEach(v => { m = Math.max(m, Math.abs(v)); }));
    const toGrid = px => { const g = []; for (let r = 0; r < 8; r++) g.push(px.slice(r * 8, r * 8 + 8)); return g; };
    DIG = { S, m, imgs: D.slice(0, 6).map(d => toGrid(d.pixels)) };
    return DIG;
  }

  /* ---------- the points that are on screen ---------- */
  function stdScale(s) {
    const Sc = covN(s.pts); const sdx = Math.sqrt(Sc[0][0]), sdy = Math.sqrt(Sc[1][1]);
    return [LA.lerp(1, 1 / sdx, s.stdT), LA.lerp(1, 1 / sdy, s.stdT)];
  }
  function displayPts(s) {
    if (s.src === 'gen') { const L = chol2(s.sx, s.sy, s.rho); return s.zp.map(z => N.add(s.gmu, N.matVec(L, z))); }
    if (s.src === 'sample') {
      const u1 = LA.clamp(s.sampleT, 0, 1), u2 = LA.clamp(s.sampleT - 1, 0, 1);
      const M = lerpM(I2(), L0, u1), off = N.scale(MU, u2);
      return s.z50.map(z => N.add(off, N.matVec(M, z)));
    }
    if (s.src === 'three') return [[-1, -1], [0, 0], [1, 1]];
    const sc = stdScale(s), sh = N.scale(s.mu, s.centerT);
    return s.pts.map(p => [(p[0] - sh[0]) * sc[0], (p[1] - sh[1]) * sc[1]]);
  }
  function stats(P) { const mu = N.mean(P); const S = covN(P); return { mu, S, e: N.eigSym(S), H: sqrtSym(S) }; }

  /* ---------- screen tile Σ = [a b; c d] ---------- */
  function tile(g, S, o) {
    o = o || {};
    const x0 = o.x === undefined ? g.size.w - 230 : o.x, y0 = o.y === undefined ? 30 : o.y;
    g.label(o.name || 'Σ =', [x0, y0 + 13], { screen: true, anchor: 'left', size: 17, color: 'hud', font: 'math' });
    g.label('[', [x0 + 46, y0 + 12], { screen: true, size: 48, color: 'muted', font: 'ui', bg: false });
    g.label(']', [x0 + 176, y0 + 12], { screen: true, size: 48, color: 'muted', font: 'ui', bg: false });
    const cx = [x0 + 84, x0 + 142], cy = [y0, y0 + 26];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      const show = o.cells ? o.cells(i, j) : true;
      const col = o.colorAt ? o.colorAt(i, j) : (i === j ? (i === 0 ? 'i' : 'j') : 'b');
      g.label(show ? LA.fmt.fixed(S[i][j], 2) : '·', [cx[j], cy[i]], { screen: true, size: 15.5, color: col, font: 'mono', bg: false, bold: o.glow && o.glow(i, j) });
    }
    return (i, j) => [cx[j], cy[i]];
  }

  LA.registerLesson({
    id: 'p4-09-covariance',
    part: 4,
    title: 'Mean, centering, covariance as a shape; Gaussians',
    subtitle: 'Center the cloud, measure Σ, draw its ellipse with gold axes — and build the cloud from round noise as μ + Lz.',
    minutes: 11,
    prereqs: ['p3-07-gram-matrix', 'p4-06-symmetric-quadratic-forms', 'p1-07-data-as-vectors'],
    tags: ['core'],
    badge: { drawnIn: '2D cloud + one 3D ellipsoid', realSize: 'a covariance of 768 embedding dimensions is 768×768; a Gaussian over 784 pixels has a 784×784 Σ', transfers: 'Σ = XcᵀXc/n, symmetric PSD, eigenvectors = axes, x = μ + Lz — identical in any dimension' },
    cantShow: ['the ellipsoid for d > 3 (a 64×64 Σ is shown as a heatmap instead)', 'heavy tails or non-Gaussian clouds: the ellipse only summarizes second moments, whatever the shape'],
    terms: [
      { term: 'mean', def: 'The average point of a cloud: its center, written μ.' },
      { term: 'centering', def: 'Subtracting the mean from every point so the cloud sits at the origin.' },
      { term: 'variance', def: 'The average squared spread of one feature around its mean.' },
      { term: 'covariance', def: 'The average product of two features’ deviations: positive when they rise together, negative when one rises as the other falls.' },
      { term: 'covariance matrix', def: 'Σ: variances on the diagonal, covariances off it; for centered data Σ = XcᵀXc/n.' },
      { term: 'Gaussian', def: 'The bell-shaped cloud whose one-sigma contour is the ellipse of Σ.' },
      { term: 'Cholesky factor', def: 'A lower-triangular “square root” L of Σ with LLᵀ = Σ, used to generate samples μ + Lz.' },
    ],
    atlas: ['gaussian', 'pca', 'normalization'],
    thumbnail: 4,

    state() {
      const z49 = unitNoise(49, 80, 2);
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), { center: [1.2, 0.5], halfH: 3.9 }),
        pts: z49.map(z => N.add(MU, N.matVec(L0, z)).map(x => Math.round(x * 1e4) / 1e4)),
        mu: MU.slice(), z49, zp: LA.clone(z49), z50: unitNoise(50, 80, 2),
        p3: unitNoise(50, 150, 3).map(z => N.matVec(L3, z).map(x => Math.round(x * 1e3) / 1e3)),
        scene: 'main', src: 'data', nShow: 80, centerT: 0, stdT: 0, ghostOrig: false, muVec: 0,
        varX: 0, varY: 0, covT: 0, tileT: 0, ellT: 0, axesT: 0, rings: false, fillDet: 0, cross: false,
        sampleT: 0, gmu: [0, 0], sx: Math.sqrt(2), sy: 1, rho: RHO0, genHandles: false, pointHandles: false,
        marker: null, Sig: [[0, 0], [0, 0]], n3: 150, mlT: 0, panel: 'mu',
      };
    },

    render(g, s, ctx) {
      if (s.scene === '3d' && s.view === '3d') { render3d(g, s); return; }
      if (s.scene === 'heat') { renderHeat(g, s); return; }
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      if (s.view === '3d') { render3d(g, s); return; }
      renderMain(g, s, ctx);
    },

    math(s, tex) {
      const P = []; const mu = tex.sym('μ'); const tn = (x, o) => tex.text(LA.fmt(x), { color: typeof o === 'string' ? o : (o && o.color) });
      const Sig = tex.sym('Σ');
      const live = () => stats(displayPts(s));
      const cell = (x, c) => tn(x, c);
      switch (s.panel) {
        case 'mu': {
          const m = N.mean(s.pts);
          P.push(tex.eq(mu, tex.op('='), tex.frac('1', 'n'), tex.sum(tex.sym('x', { sub: 'i' }), 'i', ''), tex.op('='), tex.vec(m)));
          P.push(tex.text('n = 80 synthetic points (LA.rng seed 49)', { color: 'muted' }));
          break; }
        case 'center': {
          const st = live();
          P.push(tex.eq(tex.sym('X', { sub: 'c' }), tex.op('='), tex.sym('X'), tex.op('−'), mu));
          P.push(tex.eq(tex.text('mean of the cloud on screen'), tex.op('='), tex.vec(st.mu.map(x => Math.round(x * 100) / 100))));
          break; }
        case 'var': {
          const st = live();
          P.push(tex.eq(tex.sym('Σ', { sub: '11' }), tex.op('='), tex.frac('1', 'n'), tex.sum(tex.sym('x', { sub: 'i', sup: '2' }), 'i', ''), tex.op('='), cell(st.S[0][0], 'i')));
          P.push(tex.eq(tex.sym('Σ', { sub: '22' }), tex.op('='), tex.frac('1', 'n'), tex.sum(tex.sym('y', { sub: 'i', sup: '2' }), 'i', ''), tex.op('='), cell(st.S[1][1], 'j')));
          break; }
        case 'cov': {
          const st = live();
          P.push(tex.eq(tex.sym('Σ', { sub: '12' }), tex.op('='), tex.sym('Σ', { sub: '21' }), tex.op('='), tex.frac('1', 'n'), tex.sum(tex.sym('x', { sub: 'i' }) + tex.sym('y', { sub: 'i' }), 'i', ''), tex.op('='), cell(st.S[0][1], 'b')));
          P.push(tex.eq(Sig, tex.op('='), tex.mat(st.S, { colors: (i, j) => i === j ? (i ? 'j' : 'i') : 'b' }), tex.op('='), tex.T(tex.sym('X', { sub: 'c' })), tex.sym('X', { sub: 'c' }), tex.op('/'), tex.sym('n')));
          break; }
        case 'ell': {
          const st = live();
          P.push(tex.eq(Sig, tex.op('='), tex.sym('Q'), tex.sym('Λ'), tex.T(tex.sym('Q'))));
          P.push(tex.eq(tex.text('axes'), tex.op('='), tex.sqrt(tex.sym('λ', { sub: 'i' })), tex.sym('q', { sub: 'i', color: 'gold' })));
          P.push(tex.eq(tex.sym('λ'), tex.op('='), cell(st.e.values[0], 'gold'), tex.op(','), cell(st.e.values[1], 'gold'), tex.op(';'), tex.sqrt(tex.sym('λ')), tex.op('='), cell(Math.sqrt(st.e.values[0]), 'gold'), tex.op(','), cell(Math.sqrt(Math.max(0, st.e.values[1])), 'gold')));
          break; }
        case 'std': {
          const st = live(); const r = st.S[0][1] / Math.sqrt(st.S[0][0] * st.S[1][1]);
          P.push(tex.eq(tex.sym('Σ', { sub: 'std' }), tex.op('='), tex.mat(st.S, { colors: (i, j) => i === j ? (i ? 'j' : 'i') : 'bad' })));
          P.push(tex.eq(tex.sym('ρ'), tex.op('='), cell(r, 'bad'), tex.text('the tilt survives', { color: 'bad' })));
          break; }
        case 'sample':
          P.push(tex.eq(tex.sym('z'), tex.op('~'), tex.text('N(0, I)')));
          P.push(tex.eq(tex.sym('x'), tex.op('='), mu, tex.op('+'), tex.sym('L'), tex.sym('z')));
          P.push(tex.eq(tex.sym('L'), tex.op('='), tex.mat(L0), tex.op(','), tex.sym('L'), tex.T(tex.sym('L')), tex.op('='), Sig));
          break;
        case 'det': {
          const S = genSigma(s.sx, s.sy, s.rho); const d = Math.max(0, N.det(S));
          P.push(tex.eq(tex.text('det'), Sig, tex.op('='), tn(S[0][0]), tex.op('·'), tn(S[1][1]), tex.op('−'), tn(S[0][1]), tex.op('²'), tex.op('='), cell(d, d < 0.05 ? 'bad' : 'gold')));
          P.push(tex.eq(tex.text('area'), tex.op('='), tex.text('π'), tex.sqrt(tex.text('det') + Sig), tex.op('='), cell(Math.PI * Math.sqrt(d), d < 0.05 ? 'bad' : 'gold')));
          break; }
        case '3d':
          P.push(tex.eq(tex.sym('Σ', { sub: '3' }), tex.op('='), tex.mat(S3), tex.op('='), tex.sym('Q'), tex.sym('Λ'), tex.T(tex.sym('Q'))));
          P.push(tex.eq(tex.sqrt(tex.sym('λ')), tex.op('='), cell(Math.sqrt(E3.values[0]), 'gold'), tex.op(','), cell(Math.sqrt(E3.values[1]), 'gold'), tex.op(','), cell(Math.sqrt(E3.values[2]), 'gold')));
          break;
        case 'heat':
          P.push(tex.eq(Sig, tex.op('∈'), tex.text('ℝ⁶⁴ˣ⁶⁴'), tex.op(','), Sig, tex.op('='), tex.T(tex.sym('X', { sub: 'c' })), tex.sym('X', { sub: 'c' }), tex.op('/'), tex.sym('n')));
          P.push(tex.text(HONEST));
          break;
        case 'ml':
          P.push(tex.eq(tex.sym('x'), tex.op('='), mu, tex.op('+'), tex.sym('L'), tex.sym('z')));
          break;
        case 'recap':
          P.push(tex.eq(Sig, tex.op('='), tex.T(tex.sym('X', { sub: 'c' })), tex.sym('X', { sub: 'c' }), tex.op('/'), tex.sym('n')));
          break;
        case 'marker':
          P.push(tex.eq(tex.text('marker'), tex.op('='), tex.vec(s.marker || [0, 0], { color: 'muted' })));
          break;
        case 'three': {
          const S = s.Sig;
          P.push(tex.eq(tex.text('points'), tex.rowvec([-1, -1]), tex.rowvec([0, 0]), tex.rowvec([1, 1])));
          P.push(tex.eq(Sig, tex.op('='), tex.mat(S, { colors: 'teal' }), tex.op(','), tex.text('det'), tex.op('='), tn(N.det(S))));
          break; }
        default: { // play, gen
          const st = live();
          P.push(tex.eq(mu, tex.op('='), tex.vec(st.mu.map(x => Math.round(x * 100) / 100)), Sig, tex.op('='), tex.mat(st.S, { colors: (i, j) => i === j ? (i ? 'j' : 'i') : 'b' })));
          P.push(tex.eq(tex.sym('λ'), tex.op('='), cell(st.e.values[0], 'gold'), tex.op(','), cell(st.e.values[1], 'gold'), tex.op(';'), tex.text('det'), Sig, tex.op('='), tn(N.det(st.S))));
          const L = N.chol(st.S);
          P.push(tex.eq(tex.sym('L'), tex.op('='), L ? tex.mat(L) : tex.text('— (det Σ = 0)', { color: 'bad' })));
        }
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'A cloud and its center', duration: 3,
        narration: 'Eighty synthetic data points with two features, drawn as a tilted cloud of blue dots. First question: where is its center? Average all the x’s and all the y’s: the <b>mean</b> μ = [2, 1]. One point, drawn as the white arrow μ, stands for the whole cloud.',
        setup(s) { reset(s, { panel: 'mu', nShow: 0, muVec: 0 }); s.cam = cam2(1.0, 0.6, 4.9); },
        at(s, t) { s.nShow = Math.round(80 * seg(t, 0, 0.6, 'linear')); s.muVec = seg(t, 0.62, 1); } },

      { title: 'Centering: slide the cloud to the origin', duration: 3,
        narration: '<b>Centering</b>: subtract μ from every point. The cloud slides until its center sits on the origin; its shape is untouched, and a faint ghost stays behind. Centered data makes every later formula simpler: from here on, “spread” is just “distance from the origin”.',
        setup(s) { reset(s, { panel: 'center', muVec: 1, ghostOrig: true }); s.cam = cam2(1.0, 0.6, 4.9); },
        at(s, t) { s.centerT = seg(t, 0.1, 0.9); s.muVec = 1 - s.centerT; } },

      { title: 'Variance along each axis', duration: 4,
        narration: 'How wide is the cloud left-to-right? Square each x-deviation (the green squares) and average: the <b>variance</b> of feature 1 is 2.0 — the area of the average square. Up-and-down, the red squares average to 1.0. Two numbers for two axes — but they miss the tilt.',
        setup(s) { reset(s, { panel: 'var', centerT: 1 }); s.cam = cam2(1.0, 0.6, 4.9); },
        at(s, t) { s.varX = seg(t, 0, 0.5, 'linear'); s.varY = seg(t, 0.5, 1, 'linear'); LA.cam.tween(s, { center: [1.0, 0.6], halfH: 4.9 }, { center: [0, 0], halfH: 4.0 }, seg(t, 0, 0.15)); } },

      { title: 'Covariance: do they move together?', duration: 4.5,
        narration: 'Multiply each point’s x by its y and average. Points top-right and bottom-left give positive products (orange), the other quadrants negative (blue). The cloud leans up-right, so the average is positive: <b>covariance</b> = 1.2. All three numbers form the symmetric <b>covariance matrix</b> Σ = XcᵀXc/n — the Gram matrix of the two feature columns.',
        setup(s) { reset(s, { panel: 'cov', centerT: 1 }); s.cam = cam2(0, 0, 4.0); },
        at(s, t) { s.covT = t; s.tileT = seg(t, 0.72, 1, 'linear'); } },

      { title: 'The ellipse of Σ', duration: 4.5,
        narration: 'Σ is symmetric, so p4-06 applies: it has perpendicular gold eigenvectors ' + goldS('q₁') + ', ' + goldS('q₂') + ' and turns the unit circle into an ellipse. Scaled by the square roots of the eigenvalues, the teal ellipse hugs the cloud. Its long axis is the direction of greatest spread; the half-lengths are the spreads √λ₁ ≈ 1.67 and √λ₂ ≈ 0.45.',
        setup(s) { reset(s, { panel: 'ell', centerT: 1, tileT: 1 }); s.cam = cam2(0, 0, 4.0); },
        at(s, t) { s.ellT = seg(t, 0.05, 0.6); s.axesT = seg(t, 0.55, 0.95); } },

      { title: 'Misconception: scaling each feature removes the tilt', duration: 4, misconception: true,
        narration: '✕ “Divide x by its spread and y by its spread, and the features become independent.” Do it: the ellipse changes shape but stays tilted. Standardizing sets the diagonal to 1; the off-diagonal becomes the correlation 0.85 and stays. Un-tilting needs a rotation — whitening, in p4-11.',
        setup(s) { reset(s, { panel: 'std', centerT: 1, tileT: 1, ellT: 1, axesT: 1 }); s.cam = cam2(0, 0, 4.0); },
        at(s, t) { s.stdT = seg(t, 0.05, 0.6); s.cross = t > 0.65; } },

      { title: 'Making the cloud: x = μ + Lz', duration: 5,
        narration: 'Run it backwards. Start from round noise z: 80 gray points with Σ = I. Multiply by the <b>Cholesky factor</b> L, a square root of Σ with LLᵀ = Σ (here L = [[1.41, 0], [0.85, 0.53]]): the round cloud becomes the tilted one. Add μ and it slides over the original’s ghost. Every <b>Gaussian</b> is a stretched, shifted ball of unit noise.',
        setup(s) { reset(s, { panel: 'sample', src: 'sample', ghostOrig: true, ellT: 1, sampleT: 0 }); s.cam = cam2(0, 0, 4.0); },
        at(s, t) { s.sampleT = 2 * seg(t, 0.08, 0.88, 'linear'); s.rings = t > 0.9; LA.cam.tween(s, { center: [0, 0], halfH: 4.0 }, { center: [1.2, 0.7], halfH: 4.9 }, seg(t, 0.45, 0.85)); } },

      { title: 'det Σ: how much room the cloud takes', duration: 4,
        narration: 'The 1σ ellipse has area π·√det Σ = π·√0.56 ≈ 2.35. Now push the correlation to 1: the features become perfectly dependent, the cloud collapses onto a line, det Σ → 0 and the ellipse has no area at all. That determinant normalizes the Gaussian’s bell, and det 0 is why duplicated features break Gaussian models.',
        setup(s) { reset(s, { panel: 'det', src: 'gen', gmu: [0, 0], sx: Math.sqrt(2), sy: 1, rho: RHO0, ellT: 1, axesT: 1, tileT: 1 }); s.zp = LA.clone(s.z49); s.cam = cam2(0, 0, 4.0); },
        at(s, t) { s.fillDet = seg(t, 0, 0.2); s.rho = LA.lerp(RHO0, 1, seg(t, 0.4, 0.9)); } },

      { title: 'In 3D: an ellipsoid with three axes', duration: 4.5,
        narration: 'Three features give a 3×3 Σ and an ellipsoid with three perpendicular gold axes. This cloud is long along ' + goldS('q₁') + ' (√λ = 3), medium along ' + goldS('q₂') + ' (1.2) and very thin along ' + goldS('q₃') + ' (0.3) — nearly a pancake. Drag to orbit. PCA, next, decides which axes to keep.',
        setup(s) { reset(s, { panel: '3d', scene: '3d', n3: 0 }); s.view = '3d'; s.cam = cam3(); },
        at(s, t) { s.n3 = Math.round(150 * seg(t, 0, 0.45, 'linear')); LA.cam.tween(s, { yaw: 0.61 }, { yaw: 1.25 }, seg(t, 0, 1, 'linear')); } },

      { title: '64 features: a heatmap, not a shape', duration: 3.5,
        narration: 'For the 64 pixel features of synthetic 8×8 digits, Σ is a 64×64 heatmap: a bright diagonal (variances) and structured off-diagonals (pixels that light up together). The picture stops at 3. The formula does not — Σ = XcᵀXc/n with more terms, and its eigenvectors are still the directions of spread.',
        setup(s) { reset(s, { panel: 'heat', scene: 'heat', mlT: 0 }); s.view = '2d'; s.cam = cam2(0, 0, 3.6); },
        at(s, t) { s.mlT = seg(t, 0.05, 0.6); } },

      { title: 'Why ML cares', duration: 4, mlTie: true,
        narration: 'PCA reads Σ’s axes to compress data (next lesson). Mahalanobis distance and whitening use Σ to measure distance in the cloud’s own units (p4-11). Variational autoencoders sample exactly x = μ + Lz so gradients can flow through randomness — the reparameterization trick. And feature scaling is just taming Σ’s diagonal.',
        setup(s) { reset(s, { panel: 'ml', scene: 'ml', mlT: 0 }); s.view = '2d'; s.cam = cam2(0, 0, 3.6); },
        at(s, t) { s.mlT = seg(t, 0.1, 0.8); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Center at μ. Σ holds variances on the diagonal and covariance off it: Σ = XcᵀXc/n. Its gold eigen-axes and √λ draw the ellipse that hugs the cloud; det Σ measures its area. Any Gaussian is unit noise stretched by L and shifted by μ.',
        cards: ['<b>Center, then measure.</b> Subtract μ; the cloud’s shape is untouched.', '<b>Σ’s gold axes draw the ellipse.</b> Eigenvectors give its directions, √λ its half-lengths.', '<b>Gaussian = μ + Lz.</b> Round noise, stretched by L (LLᵀ = Σ), shifted by μ.'],
        setup(s) { reset(s, { panel: 'recap', scene: 'recap' }); s.view = '2d'; s.cam = cam2(0, 0, 3.6); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>The cloud is <b>μ + Lz</b> for a fixed noise sample z. Slide the spreads and the correlation (or drag the white handles: σx, σy and the ρ track), drag any blue point, and watch μ, Σ, the teal ellipse and its gold axes update live.</p>',
      setup(s) { reset(s, { panel: 'play', src: 'gen', gmu: MU.slice(), sx: Math.sqrt(2), sy: 1, rho: RHO0, ellT: 1, axesT: 1, tileT: 1, rings: true, genHandles: true, pointHandles: true }); s.zp = LA.clone(s.z49); s.cam = cam2(2, 0.3, 4.6); },
      controls: [
        { type: 'slider', bind: 'sx', min: 0.2, max: 3, step: 0.01, label: 'spread x (σx)', color: 'i' },
        { type: 'slider', bind: 'sy', min: 0.2, max: 3, step: 0.01, label: 'spread y (σy)', color: 'j' },
        { type: 'slider', bind: 'rho', min: -0.99, max: 0.99, step: 0.01, label: 'correlation ρ', color: 'b' },
        { type: 'toggle', bind: 'rings', label: '1σ / 2σ rings' },
        { type: 'preset', label: 'Clouds', items: [
          { label: 'Seed-49 cloud', apply(s) { toPlay2d(s); Object.assign(s, { sx: Math.sqrt(2), sy: 1, rho: RHO0 }); s.zp = LA.clone(s.z49); } },
          { label: 'Round (ρ = 0)', apply(s) { toPlay2d(s); Object.assign(s, { sx: 1, sy: 1, rho: 0 }); s.zp = LA.clone(s.z49); } },
          { label: 'Nearly flat (ρ = 0.98)', apply(s) { toPlay2d(s); Object.assign(s, { sx: 1.4, sy: 1, rho: 0.98 }); s.zp = LA.clone(s.z49); } },
          { label: 'Anti-correlated (ρ = −0.8)', apply(s) { toPlay2d(s); Object.assign(s, { sx: 1.4, sy: 1, rho: -0.8 }); s.zp = LA.clone(s.z49); } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2D cloud', apply(s) { toPlay2d(s); } },
          { label: '3D ellipsoid', apply(s) { s.view = '3d'; s.scene = '3d'; s.n3 = 150; s.cam = cam3(); s.panel = '3d'; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the gray marker to the mean of the cloud.',
        setup(s) { reset(s, { panel: 'marker', marker: [0, 0] }); s.cam = cam2(1.6, 0.8, 4.0); },
        judge(s) { return LA.judge.near(s.marker, N.mean(s.pts), LA.TOL.pos); },
        hint: 'Balance point: as many points to its left as to its right, as many above as below.',
        solution(s) { s.marker = N.mean(s.pts); } },
      { kind: 'entry',
        prompt: 'Type the covariance matrix Σ (divide by n = 3) of the three centered points (−1, −1), (0, 0), (1, 1). The teal ellipse follows what you type.',
        bind: 'Sig', answer: [[0.67, 0.67], [0.67, 0.67]], tol: LA.TOL.entry, label: 'Σ', colors: 'teal',
        setup(s) { reset(s, { panel: 'three', src: 'three', ellT: 1 }); s.Sig = [[0, 0], [0, 0]]; s.cam = cam2(0, 0, 2.2); } },
      { kind: 'manipulate',
        prompt: 'Make the ellipse a circle: drag the white handles (spread x, spread y and the ρ track).',
        setup(s) { reset(s, { panel: 'gen', src: 'gen', gmu: [0, 0], sx: 1.5, sy: 0.8, rho: 0.6, ellT: 1, axesT: 1, tileT: 1, genHandles: true }); s.zp = LA.clone(s.z49); s.cam = cam2(0, -0.6, 4.0); },
        judge(s) { const S = genSigma(s.sx, s.sy, s.rho); const off = Math.abs(S[0][1]), dd = Math.abs(S[0][0] - S[1][1]); return { ok: off < 0.05 && dd < 0.1, progress: Math.max(0, 1 - (off + dd) / 1.6) }; },
        hint: 'No tilt (ρ = 0) and equal spreads (σx = σy).',
        solution(s) { s.sx = 1; s.sy = 1; s.rho = 0; } },
    ],
  });

  /* ---------- step helpers ---------- */
  function cam2(x, y, h) { return Object.assign(LA.cam.defaults2d(), { center: [x, y], halfH: h }); }
  function cam3() { return Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0], dist: 17, yaw: 0.61, pitch: 0.42 }); }
  function reset(s, o) {
    Object.assign(s, {
      scene: 'main', view: '2d', src: 'data', nShow: 80, centerT: 0, stdT: 0, ghostOrig: false, muVec: 0,
      varX: 0, varY: 0, covT: 0, tileT: 0, ellT: 0, axesT: 0, rings: false, fillDet: 0, cross: false,
      sampleT: 0, genHandles: false, pointHandles: false, marker: null, n3: 150, mlT: 0,
    }, o || {});
  }
  function toPlay2d(s) { s.view = '2d'; s.scene = 'main'; s.src = 'gen'; s.gmu = MU.slice(); s.panel = 'play'; s.cam = cam2(2, 0.3, 4.6); }

  /* ---------- main 2D scene ---------- */
  function renderMain(g, s, ctx) {
    const P = displayPts(s);
    g.ghostGrid(); g.axes();
    if (s.covT > 0) drawCov(g, P, s.covT);
    if (s.ghostOrig) g.points(s.pts, { r: 2.6, color: 'muted', alpha: 0.3 });
    if (s.varX > 0) drawVar(g, P, 0, s.varX);
    if (s.varY > 0) drawVar(g, P, 1, s.varY);
    const nShow = Math.min(P.length, Math.round(s.nShow));
    const pc = s.src === 'sample' ? C.mix(C.muted, C.data, LA.clamp(s.sampleT, 0, 1)) : 'data';
    if (s.src === 'three') g.points(P, { r: 6, color: 'data' });
    else g.points(P.slice(0, nShow), { r: 3.6, color: pc });
    const st = stats(P);
    // the ellipse of Σ (or of the typed Σ in check 2)
    if (s.ellT > 0) {
      let H = st.H, c = st.mu;
      if (s.src === 'three') { const Ss = [[s.Sig[0][0], (s.Sig[0][1] + s.Sig[1][0]) / 2], [(s.Sig[0][1] + s.Sig[1][0]) / 2, s.Sig[1][1]]]; H = sqrtSym(Ss); c = [0, 0]; }
      if (s.src === 'sample') { const u1 = LA.clamp(s.sampleT, 0, 1), u2 = LA.clamp(s.sampleT - 1, 0, 1); H = lerpM(I2(), L0, u1); c = N.scale(MU, u2); }
      const M = lerpM(I2(), H, s.ellT);
      const d = Math.abs(N.det(H));
      const flatBad = s.src === 'gen' && s.fillDet > 0 && d < 0.12;
      if (s.fillDet > 0) g.ellipse(M, c, { fill: flatBad ? 'bad' : 'gold', alpha: 0.22 * s.fillDet, stroke: null });
      g.ellipse(M, c, { stroke: flatBad ? 'bad' : 'teal', width: 2.2 });
      // label at the ellipse's left-most point
      let ua = [1, 0], best = Infinity;
      for (let k = 0; k < 36; k++) { const a = k / 36 * 2 * Math.PI, p = N.matVec(M, [Math.cos(a), Math.sin(a)]); if (p[0] < best) { best = p[0]; ua = [Math.cos(a), Math.sin(a)]; } }
      if (s.ellT > 0.9 && s.src !== 'three') g.label('1σ', N.add(c, N.matVec(M, ua)), { color: 'teal', size: 13, dx: -16, font: 'ui' });
      if (s.rings) { g.ellipse(N.scaleM(M, 2), c, { stroke: 'teal', width: 1.3, dashed: true, strokeAlpha: 0.7 }); g.label('2σ', N.add(c, N.matVec(N.scaleM(M, 2), ua)), { color: 'teal', size: 12, dx: -16, font: 'ui' }); }
      if (s.fillDet > 0) {
        const det = N.det(st.S);
        g.label(det < 0.005 ? 'det Σ = 0: no area at all' : 'area = π√' + LA.fmt.fixed(det, 2) + ' ≈ ' + LA.fmt.fixed(Math.PI * Math.sqrt(Math.max(det, 0)), 2), [0, -2.3], { color: flatBad ? 'bad' : 'gold', size: 14, font: 'ui', bold: true });
      }
    }
    if (s.axesT > 0 && s.src !== 'three') {
      st.e.vectors.forEach((q, i) => {
        const len = Math.sqrt(Math.max(st.e.values[i], 0)) * s.axesT;
        if (len < 0.02) return;
        const tip = N.add(st.mu, N.scale(q, len));
        g.vector(st.mu, tip, { color: 'gold', width: 3, head: 11, label: 'q' + SUB[i + 1], labelFont: 'math' });
        if (s.axesT > 0.95 && s.src !== 'sample' && !s.genHandles) g.label('√λ' + SUB[i + 1] + ' = ' + LA.fmt.fixed(len, 2), N.add(st.mu, N.scale(q, len * 0.55)), { color: 'gold', size: 12.5, font: 'ui', dx: i === 0 ? 14 : -18, dy: i === 0 ? 16 : -14, anchor: i === 0 ? 'left' : 'right' });
      });
    }
    // μ arrow (steps 1–2) and readout
    if (s.muVec > 0.01 && s.src === 'data') {
      const tip = N.scale(s.mu, s.muVec);
      g.vector(tip, { color: 'white', width: 3, label: 'μ', labelFont: 'math' });
      if (s.panel === 'mu') g.label('μ = [' + LA.fmt.fixed(s.mu[0], 1) + ', ' + LA.fmt.fixed(s.mu[1], 1) + ']', tip, { color: 'white', anchor: 'left', dx: 10, dy: 24, size: 13, font: 'ui' });
    }
    if (s.centerT > 0 && s.centerT < 1 && s.src === 'data') g.label('− μ', N.scale(s.mu, 1 - s.centerT * 0.5), { color: 'white', size: 13, font: 'ui', dy: -16 });
    // sampling captions
    if (s.src === 'sample') {
      const stage = s.sampleT < 0.02 ? 'z: round unit noise, Σ = I' : s.sampleT < 1 ? 'Lz: stretching the ball…' : s.sampleT < 1.98 ? 'μ + Lz: sliding into place…' : 'μ + Lz lands on the original (ghost)';
      g.hud(stage, 16, 24, { size: 14, color: 'hud', bold: true });
    }
    // Σ tile
    if (s.tileT > 0) {
      const tt = s.tileT, isStd = s.panel === 'std';
      const cellAt = tile(g, st.S, {
        cells: (i, j) => s.covT > 0 ? (i === j || tt > 0.35) : true,
        colorAt: (i, j) => i === j ? (i ? 'j' : 'i') : (isStd && s.cross ? 'bad' : 'b'),
        glow: (i, j) => i !== j && s.covT > 0 && tt > 0.35 && tt < 0.8,
      });
      if (s.cross && isStd) { const p = cellAt(0, 1); g.marker.cross([p[0] + 40, p[1]], { screen: true, size: 22 }); g.label('still correlated', [p[0] - 20, p[1] + 52], { screen: true, color: 'bad', size: 13, font: 'ui' }); }
    }
    if (s.cross && s.panel === 'std') {
      g.label('“scaled ⇒ independent”', [-1.2, 2.25], { color: 'bad', size: 14, font: 'ui' });
      g.marker.cross([-1.2, 2.25], { dx: 108, size: 20 });
    }
    // check 1: the gray marker
    if (s.marker) { g.point(s.marker, { r: 7, color: 'muted', stroke: 'white', label: 'mean?', labelColor: 'hud' }); g.handle('m', s.marker, (p, st2) => { st2.marker = [p[0], p[1]]; }, { color: 'white' }); }
    // generator handles: σx, σy and a ρ track
    if (s.genHandles) drawGenHandles(g, s);
    if (s.pointHandles && ctx.mode === 'play' && s.src === 'gen') {
      const L = chol2(s.sx, s.sy, s.rho), Li = N.inv(L);
      if (Li) P.forEach((p, i) => {
        g.handle('p' + i, p, (q, st2) => { st2.zp[i] = N.matVec(Li, N.sub([q[0], q[1]], st2.gmu)); }, { show: false, r: 7 });
        if (ctx.hot === 'p' + i) g.marker.ring(p, { r: 9, color: 'white' });
      });
    }
  }

  function drawGenHandles(g, s) {
    const o = s.gmu, ty = o[1] - 3.3;
    const hx = [o[0] + s.sx, o[1]], hy = [o[0], o[1] + s.sy], hr = [o[0] + 2 * s.rho, ty];
    g.line([o[0] - 2, ty], [o[0] + 2, ty], { color: 'muted', width: 2, alpha: 0.7 });
    [-2, 0, 2].forEach(k => g.line([o[0] + k, ty - 0.08], [o[0] + k, ty + 0.08], { color: 'muted', width: 1.5 }));
    g.label('ρ = ' + LA.fmt.fixed(s.rho, 2), [o[0] + 2.15, ty], { anchor: 'left', color: 'b', size: 13, font: 'ui' });
    g.label('−1', [o[0] - 2, ty], { anchor: 'below', dy: 6, color: 'muted', size: 11, font: 'ui' });
    g.label('+1', [o[0] + 2, ty], { anchor: 'below', dy: 6, color: 'muted', size: 11, font: 'ui' });
    g.point(hx, { r: 4, color: 'i' }); g.label('σx', hx, { color: 'i', size: 13, dy: 16, font: 'math' });
    g.point(hy, { r: 4, color: 'j' }); g.label('σy', hy, { color: 'j', size: 13, dx: -18, font: 'math' });
    g.point(hr, { r: 4, color: 'b' });
    g.handle('sx', hx, (p, st) => { st.sx = LA.clamp(Math.abs(p[0] - st.gmu[0]), 0.2, 3); }, { color: 'white' });
    g.handle('sy', hy, (p, st) => { st.sy = LA.clamp(Math.abs(p[1] - st.gmu[1]), 0.2, 3); }, { color: 'white' });
    g.handle('rho', hr, (p, st) => { st.rho = LA.clamp((p[0] - st.gmu[0]) / 2, -0.99, 0.99); }, { color: 'white' });
  }

  function drawVar(g, P, axis, u) {
    const a = seg(u, 0, 0.35), m = seg(u, 0.45, 0.95);
    const role = axis === 0 ? 'i' : 'j';
    let v = 0; P.forEach(p => { v += p[axis] * p[axis]; }); v /= P.length;
    const r = Math.sqrt(v);
    if (m < 1) P.forEach(p => {
      const d = p[axis], side = LA.lerp(Math.abs(d), r, m) * Math.sign(d || 1);
      const foot = axis === 0 ? [p[0], 0] : [0, p[1]];
      g.line(p, foot, { color: role, dashed: [3, 4], alpha: 0.4 * a * (1 - m), width: 1 });
      const sq = axis === 0 ? [[0, 0], [side, 0], [side, -Math.abs(side)], [0, -Math.abs(side)]] : [[0, 0], [0, side], [-Math.abs(side), side], [-Math.abs(side), 0]];
      g.polygon(sq, { fill: role, alpha: 0.05 * a * (1 - m) });
    });
    if (m > 0) {
      const sq = axis === 0 ? [[0, 0], [r, 0], [r, -r], [0, -r]] : [[0, 0], [0, r], [-r, r], [-r, 0]];
      g.polygon(sq, { fill: role, alpha: 0.32 * m, stroke: role, width: 2 });
      const c = axis === 0 ? [r / 2, -r / 2] : [-r / 2, r / 2];
      g.label((axis === 0 ? 'var(x) = ' : 'var(y) = ') + LA.fmt.fixed(v, 2), c, { color: role, size: 13.5, font: 'ui', bold: true, alpha: m });
      g.label('area = average square', c, { color: 'muted', size: 11.5, font: 'ui', dy: 17, alpha: m });
    }
  }

  function drawCov(g, P, u) {
    const a = seg(u, 0, 0.12), E = 8;
    g.polygon([[0, 0], [E, 0], [E, E], [0, E]], { fill: 'b', alpha: 0.07 * a });
    g.polygon([[0, 0], [-E, 0], [-E, -E], [0, -E]], { fill: 'b', alpha: 0.07 * a });
    g.polygon([[0, 0], [-E, 0], [-E, E], [0, E]], { fill: 'data', alpha: 0.07 * a });
    g.polygon([[0, 0], [E, 0], [E, -E], [0, -E]], { fill: 'data', alpha: 0.07 * a });
    if (a > 0.5) { g.label('x·y > 0', [1.5, 3.3], { color: 'b', size: 13, font: 'ui' }); g.label('x·y < 0', [-1.5, 3.3], { color: 'data', size: 13, font: 'ui' }); g.label('x·y > 0', [-1.5, -3.3], { color: 'b', size: 13, font: 'ui' }); g.label('x·y < 0', [1.5, -3.3], { color: 'data', size: 13, font: 'ui' }); }
    const k = Math.round(P.length * seg(u, 0.12, 0.72, 'linear'));
    let sum = 0;
    for (let i = 0; i < k; i++) {
      const p = P[i], pos = p[0] * p[1] > 0, cur = i === k - 1 && k < P.length;
      sum += p[0] * p[1];
      g.polygon([[0, 0], [p[0], 0], [p[0], p[1]], [0, p[1]]], { fill: pos ? 'b' : 'data', alpha: cur ? 0.5 : 0.045, stroke: cur ? (pos ? 'b' : 'data') : null, width: 1.5 });
    }
    const avg = k ? sum / k : 0;
    const x = 24, y = 92;
    g.bars([Math.abs(avg)], { screen: true, x, y, w: 46, h: 120, max: 2, colors: [avg >= 0 ? 'b' : 'data'], title: 'running mean of x·y' });
    g.hud((avg < 0 ? '−' : '') + LA.fmt.fixed(Math.abs(avg), 2) + '  (' + k + ' of ' + P.length + ' points)', x + 56, y + 120 - Math.min(1, Math.abs(avg) / 2) * 120, { size: 13, color: avg >= 0 ? 'b' : 'data' });
  }

  /* ---------- 3D pancake ---------- */
  function render3d(g, s) {
    g.grid(null, { range: 6, alpha: 0.3 });
    g.axes({ length: 5.5 });
    g.points(s.p3.slice(0, Math.round(s.n3)), { r: 2.8, color: 'data', alpha: 0.85 });
    g.ellipsoid(S3H, { color: 'teal', meridians: 5 });
    E3.vectors.forEach((q, i) => {
      const L = Math.sqrt(E3.values[i]), tip = N.scale(q, L);
      g.vector([0, 0, 0], tip, { color: 'gold', width: 3.2, head: i === 2 ? 7 : 12, label: 'q' + SUB[i + 1], labelFont: 'math' });
      g.hud('q' + SUB[i + 1] + ':  √λ' + SUB[i + 1] + ' = ' + LA.fmt.fixed(L, 1), 16, 50 + 20 * i, { size: 13.5, color: 'gold' });
    });
    g.hud('150 synthetic points (LA.rng seed 50) · 1σ ellipsoid of Σ₃', 16, 24, { size: 13, color: 'muted' });
    if (s.panel === '3d' && s.genHandles) g.hud('a fixed 3D cloud: the sliders and Clouds presets shape the 2D cloud (View → 2D cloud)', 16, g.size.h - 22, { size: 12.5, color: 'hud' });
  }

  /* ---------- 64×64 covariance heatmap ---------- */
  function renderHeat(g, s) {
    const D = digits();
    const B = { x0: -2.95, x1: 1.45, y0: -1.15, y1: 3.25 };
    g.heat2d(D.S, B, { range: [-D.m, D.m], ramp: t => C.diverge(2 * t - 1), smooth: false, alpha: 0.25 + 0.75 * seg(s.mlT, 0, 0.6) });
    g.polyline([[B.x0, B.y0], [B.x1, B.y0], [B.x1, B.y1], [B.x0, B.y1]], { closed: true, color: 'line', width: 1 });
    g.label('Σ: 64 × 64', [B.x1 + 0.12, B.y1 - 0.15], { anchor: 'left', color: 'hud', size: 13.5, font: 'ui', bold: true });
    g.label('pixel × pixel', [B.x1 + 0.12, B.y1 - 0.45], { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    g.label('↘ diagonal:', [B.x1 + 0.12, B.y1 - 1.0], { anchor: 'left', color: 'teal', size: 12, font: 'ui' });
    g.label('variances', [B.x1 + 0.12, B.y1 - 1.28], { anchor: 'left', color: 'teal', size: 12, font: 'ui' });
    g.label('positive', [B.x1 + 0.12, B.y0 + 0.75], { anchor: 'left', color: 'teal', size: 12, font: 'ui' });
    g.label('negative', [B.x1 + 0.12, B.y0 + 0.4], { anchor: 'left', color: 'mag', size: 12, font: 'ui' });
    g.label('covariance', [B.x1 + 0.12, B.y0 + 0.08], { anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
    D.imgs.forEach((im, k) => {
      const x0 = -2.95 + k * 0.99;
      g.image(im, { x0, x1: x0 + 0.85, y0: -2.55, y1: -1.7 }, { alpha: seg(s.mlT, 0.2, 1) });
    });
    g.label('synthetic 8×8 digits (LA.rng) — each one is a point in ℝ⁶⁴', [0, -2.85], { color: 'muted', size: 12, font: 'ui' });
    g.hud(HONEST, 16, g.size.h - 22, { size: 14, color: 'hud', bold: true });
  }

  /* ---------- ML tie-in: three thumbnails (stacked rows) ---------- */
  function renderML(g, s) {
    const u = s.mlT;
    const q1 = E3.vectors[0], q2 = E3.vectors[1], q3 = E3.vectors[2];
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0], dist: 16, yaw: 0.9, pitch: 0.42 }) }, (g2, r) => {
      g2.plane({ point: [0, 0, 0], span: [q1, q2], extentU: 6.5, extentV: 3, color: 'gold', alpha: 0.16 });
      const pts = s.p3.slice(0, 110).map(p => N.sub(p, N.scale(q3, N.dot(p, q3) * u)));
      g2.points(pts, { r: 2.4, color: 'data' });
      g2.hud('PCA (next lesson)', r.x + 12, r.y + 22, { size: 13.5, color: 'gold', bold: true });
      g2.hud('flatten the cloud onto', r.x + 12, r.y + 42, { size: 12, color: 'muted' });
      g2.hud('Σ’s two longest axes', r.x + 12, r.y + 58, { size: 12, color: 'muted' });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-2.2, 0, 2.4) }, (g2, r) => {
      const P = s.z49.map(z => N.matVec(L0, z));
      g2.points(P, { r: 2.4, color: 'data', alpha: 0.5 });
      const H = sqrtSym(SIG);
      g2.ellipse(H, [0, 0], { stroke: 'teal', width: 2 });
      g2.ellipse(N.scaleM(H, 2), [0, 0], { stroke: 'teal', width: 1.2, dashed: true, strokeAlpha: 0.6 });
      const e = N.eigSym(SIG), r1 = 0.3 + 0.9 * u;
      g2.circle([0, 0], r1, { stroke: 'muted', width: 1, dashed: true });
      g2.point(N.scale(e.vectors[0], r1), { r: 6, color: 'v', label: 'P₁' }); g2.point(N.scale(e.vectors[1], r1), { r: 6, color: 'w', label: 'P₂' });
      g2.hud('Mahalanobis & whitening (p4-11)', r.x + 12, r.y + 22, { size: 13.5, color: 'teal', bold: true });
      g2.hud('same distance from μ,', r.x + 12, r.y + 42, { size: 12, color: 'muted' });
      g2.hud('very different rarity', r.x + 12, r.y + 58, { size: 12, color: 'muted' });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-2.2, 0, 2.4) }, (g2, r) => {
      const M = lerpM(I2(), L0, u);
      g2.points(s.z50.map(z => N.matVec(M, z)), { r: 2.4, color: C.mix(C.muted, C.data, u) });
      g2.ellipse(M, [0, 0], { stroke: 'teal', width: 2 });
      g2.hud('VAE sampling', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
      g2.hud('reparameterization:', r.x + 12, r.y + 42, { size: 12, color: 'muted' });
      g2.hud('x = μ + Lz, z = round noise', r.x + 12, r.y + 58, { size: 12, color: 'muted' });
      g2.label(u < 0.5 ? 'z' : 'Lz', [2.2, 1.6], { color: 'hud', size: 15, font: 'math' });
    });
  }

  /* ---------- recap: three stacked mini-cards ---------- */
  function renderRecap(g, s) {
    const P0 = s.pts, Pc = s.z49.map(z => N.matVec(L0, z));
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(1.4, 0.6, 2.6) }, (g2, r) => {
      g2.ghostGrid(); g2.axes({ ticks: false });
      g2.points(P0, { r: 2.2, color: 'muted', alpha: 0.4 });
      g2.points(Pc, { r: 2.6, color: 'data' });
      g2.vector(MU, [0.1, 0.05], { color: 'white', width: 2.4, head: 10, label: '−μ', labelFont: 'ui', labelAt: 0.5 });
      g2.hud('1 · Center, then measure: x − μ', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(1.6, 0, 2.6) }, (g2, r) => {
      g2.ghostGrid();
      g2.points(Pc, { r: 2.4, color: 'data', alpha: 0.7 });
      const H = sqrtSym(SIG), e = N.eigSym(SIG);
      g2.ellipse(H, [0, 0], { stroke: 'teal', width: 2 });
      e.vectors.forEach((q, i) => g2.vector(N.scale(q, Math.sqrt(e.values[i])), { color: 'gold', width: 2.6, head: 9, label: 'q' + SUB[i + 1], labelFont: 'math' }));
      tile(g2, SIG, { x: r.x + r.w * 0.62, y: r.y + r.h / 2 - 14 });
      g2.hud('2 · Σ’s gold axes draw the ellipse', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(0, 0, 2.6) }, (g2, r) => {
      const left = [-3.2, 0], right = [2.6, 0];
      g2.points(s.z50.map(z => N.add(left, N.scale(z, 0.8))), { r: 2.2, color: 'muted' });
      g2.circle(left, 0.8, { stroke: 'teal', width: 1.6 });
      g2.points(s.z50.map(z => N.add(right, N.scale(N.matVec(L0, z), 0.8))), { r: 2.2, color: 'data' });
      g2.ellipse(N.scaleM(L0, 0.8), right, { stroke: 'teal', width: 1.8 });
      g2.vector([-1.8, 0], [0.9, 0], { color: 'hud', width: 2, head: 10 });
      g2.label('μ + Lz', [-0.45, 0], { color: 'white', size: 15, font: 'math', dy: -16 });
      g2.hud('3 · Gaussian = round noise, stretched by L, shifted by μ', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
  }
})();
