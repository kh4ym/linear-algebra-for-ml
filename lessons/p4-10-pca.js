/* p4-10 — PCA finds the long axis and flattens a 3D cloud (line of greatest spread, rotate + crop, PCA = SVD, a 64-D shadow). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpM = LA.lerpM, C = LA.color;
  const HONEST = 'The picture stops at 3. The formula does not.';
  const SUB = ['₀', '₁', '₂', '₃'];
  const I2 = () => [[1, 0], [0, 1]];
  const RHO0 = 1.2 / Math.sqrt(2);
  const goldS = t => '<span class="c-gold">' + t + '</span>';
  const CLASS = ['data', 'dataB', 'dataC'];
  const DEF = { scene: 'two', src: 'data', showLine: false, lineHandle: false, sweep: 0, pcT: 0, ellT: 0, rotT: 0, flatT: 0, kT: 0, bars: null, tileT: 0, n3: 150, ghosts: true, planeT: 0, scale3: 0, stdT: 0, cross: false, okMark: false, cloud3: 'pancake', useK: false, genHandles: false, digT: 1, svdT: 0, mlT: 0 };

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
  const genSigma = (sx, sy, rho) => [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]];
  const L0 = chol2(Math.sqrt(2), 1, RHO0);
  const R3 = N.matMul(N.rot3([0, 0, 1], 35 * Math.PI / 180), N.rot3([1, 0, 0], 25 * Math.PI / 180));
  const L3 = N.matMul(R3, N.diag([3, 1.2, 0.3]));
  const L3inv = N.inv(L3);
  const round = (v, k) => v.map(x => Math.round(x * k) / k);
  /** eigen-frame of a 3×3 covariance with a proper rotation (det +1). */
  function frame3(S) {
    const e = N.eigSym(S); const q = e.vectors.map(v => v.slice());
    if (N.det(N.fromCols(q)) < 0) q[2] = N.scale(q[2], -1);
    return { q, lam: e.values.map(l => Math.max(l, 0)) };
  }
  /** rotation F(u) from I (u = 0) to Qᵀ (u = 1), by axis–angle. */
  function rotTo(Qt, u) {
    if (u <= 0) return N.identity(3);
    const c = LA.clamp((N.trace(Qt) - 1) / 2, -1, 1), ang = Math.acos(c);
    const ax = N.rotationAxis(Qt);
    if (!ax || ang < 1e-9) return N.identity(3);
    return N.rot3(ax, ang * u);
  }

  // synthetic digits (three kinds) and their PCA, computed once (deterministic) and shared by state() and render
  let DIG = null;
  function digitsData() {
    if (DIG) return DIG;
    const raw = LA.rng.syntheticDigits(50, 300, { size: 8 }).filter(d => d.label <= 2).slice(0, 90);
    const X = raw.map(d => d.pixels);
    DIG = { X, lab: raw.map(d => d.label), pc: N.pca(X, 32) };
    return DIG;
  }
  const toGrid = px => { const g = []; for (let r = 0; r < 8; r++) g.push(px.slice(r * 8, r * 8 + 8).map(v => LA.clamp(v, 0, 1))); return g; };

  /* ---------- screen tile for a 2×2 matrix ---------- */
  function tile(g, S, o) {
    o = o || {};
    const x0 = o.x === undefined ? g.size.w - 230 : o.x, y0 = o.y === undefined ? 30 : o.y;
    g.label(o.name || 'Σ =', [x0, y0 + 13], { screen: true, anchor: 'left', size: 17, color: 'hud', font: 'math' });
    g.label('[', [x0 + 46, y0 + 12], { screen: true, size: 48, color: 'muted', font: 'ui', bg: false });
    g.label(']', [x0 + 176, y0 + 12], { screen: true, size: 48, color: 'muted', font: 'ui', bg: false });
    const cx = [x0 + 84, x0 + 142], cy = [y0, y0 + 26];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) g.label(LA.fmt.fixed(Math.abs(S[i][j]) < 0.005 ? 0 : S[i][j], 2), [cx[j], cy[i]], { screen: true, size: 15.5, color: i === j ? 'gold' : 'b', font: 'mono', bg: false });
  }

  LA.registerLesson({
    id: 'p4-10-pca',
    part: 4,
    title: 'PCA finds the long axis and flattens a 3D cloud',
    subtitle: 'The gold axes of Σ are the directions of spread: rotate into them, crop the short ones, and label every shadow with its % of variance.',
    minutes: 14,
    prereqs: ['p4-09-covariance', 'p4-06-symmetric-quadratic-forms', 'p4-01-change-of-basis', 'p4-07-svd'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D → 3D → 2D shadow', realSize: 'PCA on 784-pixel images, 768-dim embeddings or 20 000 genes; here the shadow of a 64-dim cloud', transfers: 'center, take Σ’s eigenvectors (= right singular vectors of Xc), project onto the top k — identical at any d; only the cloud becomes undrawable, so we draw shadows and say so' },
    cantShow: ['the 64-dimensional digit cloud itself: only its 2-D shadow and the % of variance it keeps', 'non-linear structure (a curved manifold) that PCA, being a rotation and a crop, cannot see', 'sign and order ambiguity of components with nearly equal variance (stated; LA.num.stabilize hides flicker)'],
    terms: [
      { term: 'principal component', def: 'An eigenvector of the covariance: a direction of spread, the largest first (PC1, PC2, …).' },
      { term: 'explained variance', def: 'The share λᵢ / (λ₁ + λ₂ + …) of the total spread carried by component i.' },
      { term: 'projection onto PCs', def: 'The shadow of each point on the top-k components: new coordinates with fewer numbers.' },
      { term: 'reconstruction', def: 'The projected point mapped back to the original coordinates; it misses exactly what was dropped.' },
      { term: 'PCA = SVD', def: 'The components are the right singular vectors of the centered data matrix, and λᵢ = σᵢ²/n.' },
    ],
    atlas: ['pca', 'svd', 'embeddings'],
    thumbnail: 5,

    state() {
      const z49 = unitNoise(49, 80, 2);
      const p3 = unitNoise(50, 150, 3).map(z => round(N.matVec(L3, z), 1e3));
      const D = digitsData(), pc = D.pc, idx = 3;
      const rec = k => { let x = pc.mean.slice(); for (let j = 0; j < k; j++) x = N.add(x, N.scale(pc.components[j], pc.Z[idx][j])); return round(x, 1e3); };
      const fr = frame3(covN(p3)), sv = N.svd(p3);
      const cum = k => pc.explained.slice(0, k).reduce((a, b) => a + b, 0);
      return {
        view: '2d', cam: cam2(0, 0, 4.0),
        pts2: z49.map(z => round(N.matVec(L0, z), 1e4)), z49, p3,
        dig: {
          Z: pc.Z.map(z => round([z[0], z[1]], 1e3)), lab: D.lab.slice(), mean: round(pc.mean, 1e4),
          comps: [round(pc.components[0], 1e4), round(pc.components[1], 1e4)], expl: [pc.explained[0], pc.explained[1]],
          cum2: cum(2), cum8: cum(8), cum32: cum(32), rec: [rec(2), rec(8), rec(32)], idx,
        },
        svd: { q: fr.q.map(v => round(v, 1e4)), lam: round(fr.lam, 1e4), v: N.cols(sv.V).map(v => round(v, 1e4)), s2n: round(sv.S.map(x => x * x / p3.length), 1e4) },
        scene: 'two', src: 'data', theta: 0, sweep: 0, showLine: false, pcT: 0, ellT: 0, rotT: 0, flatT: 0, kT: 0,
        bars: null, tileT: 0, n3: 150, ghosts: true, planeT: 0, axesT: 1, scale3: 0, stdT: 0, cross: false, okMark: false,
        cloud3: 'pancake', useK: false, k: 3, sx: 1.5, sy: 0.8, rho: 0.6, genHandles: false, lineHandle: false,
        digT: 1, hotDigit: 3, svdT: 0, mlT: 0, panel: 'var',
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'svd') { renderSVD(g, s); return; }
      if (s.scene === 'digits') { renderDigits(g, s, ctx); return; }
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      if (s.view === '3d') { render3(g, s); return; }
      render2(g, s, ctx);
    },

    math(s, tex) {
      const P = []; const tn = (x, c) => tex.text(LA.fmt(x), { color: c });
      const pct = x => tex.text(Math.round(100 * x) + ' %', { color: 'gold' });
      const Sig = tex.sym('Σ'), d = tex.sym('d'), Xc = tex.sym('X', { sub: 'c' });
      const q = (i) => tex.sym('q', { sub: String(i), color: 'gold' });
      switch (s.panel) {
        case 'var': {
          const B = base2(s), S = covN(B), dd = [Math.cos(s.theta), Math.sin(s.theta)];
          P.push(tex.eq(tex.text('var(shadows)'), tex.op('='), tex.frac('1', 'n'), tex.sum(tex.paren(tex.sym('x', { sub: 'i' }) + tex.op('·') + d) + '<sup>2</sup>', 'i', ''), tex.op('='), tn(N.dot(dd, N.matVec(S, dd)), 'mag')));
          P.push(tex.eq(d, tex.op('='), tex.vec([dd[0], dd[1]]), tex.text('θ = ' + LA.fmt.deg(s.theta))));
          break; }
        case 'eig': {
          const e = N.eigSym(covN(base2(s)));
          P.push(tex.eq(tex.text('var'), tex.op('='), tex.T(d), Sig, d, tex.op(','), Sig, q(1), tex.op('='), tex.sym('λ', { sub: '1' }), q(1)));
          P.push(tex.eq(tex.sym('λ'), tex.op('='), tn(e.values[0], 'gold'), tex.op(','), tn(e.values[1], 'gold')));
          break; }
        case 'rot': {
          const e = N.eigSym(covN(base2(s))), tot = e.values[0] + e.values[1];
          P.push(tex.eq(Xc, tex.sym('Q'), tex.op(','), tex.sym('Σ′'), tex.op('='), tex.T(tex.sym('Q')), Sig, tex.sym('Q'), tex.op('='), tex.text('diag'), tex.paren(tn(e.values[0], 'gold') + tex.op(',') + tn(e.values[1], 'gold'))));
          P.push(tex.eq(tex.text('explained'), tex.op('='), pct(e.values[0] / tot), tex.op('/'), tex.text(Math.round(100 * e.values[1] / tot) + ' %', { color: 'muted' })));
          break; }
        case 'flat': {
          const e = N.eigSym(covN(base2(s))), tot = e.values[0] + e.values[1];
          P.push(tex.eq(tex.sym('z', { sub: 'i' }), tex.op('='), tex.sym('x', { sub: 'i' }), tex.op('·'), q(1)));
          P.push(tex.eq(tex.text('kept'), tex.op('='), tex.frac(tex.sym('λ', { sub: '1' }), tex.sym('λ', { sub: '1' }) + tex.op('+') + tex.sym('λ', { sub: '2' })), tex.op('='), pct(e.values[0] / tot)));
          break; }
        case '3d': {
          const f = frame3(covN(base3(s)));
          P.push(tex.eq(tex.sym('Σ', { sub: '3' }), tex.op('='), tex.sym('Q'), tex.sym('Λ'), tex.T(tex.sym('Q'))));
          P.push(tex.eq(tex.sqrt(tex.sym('λ')), tex.op('≈'), tn(Math.sqrt(f.lam[0]), 'gold'), tex.op(','), tn(Math.sqrt(f.lam[1]), 'gold'), tex.op(','), tn(Math.sqrt(f.lam[2]), 'gold')));
          break; }
        case 'crop': case 'recon': {
          const f = frame3(covN(base3(s))), tot = f.lam[0] + f.lam[1] + f.lam[2];
          if (s.panel === 'crop') { P.push(tex.eq(tex.sym('Z'), tex.op('='), Xc, tex.sym('Q'), tex.op(','), tex.text('keep columns 1..2'))); }
          else { P.push(tex.eq(tex.sym('x̂', { sub: 'i' }), tex.op('='), tex.sym('Q', { sub: 'k' }), tex.T(tex.sym('Q', { sub: 'k' })), tex.sym('x', { sub: 'i' }), tex.op(','), tex.text('error'), tex.op('='), tex.sum(tex.sym('λ', { sub: 'i' }), 'i>k', ''))); }
          P.push(tex.eq(tex.text('explained'), tex.op('='), pct(f.lam[0] / tot), tex.op(','), pct(f.lam[1] / tot), tex.op(','), tex.text(Math.round(100 * f.lam[2] / tot) + ' %', { color: 'muted' })));
          break; }
        case 'units': {
          const f = frame3(covN(base3(s)));
          P.push(tex.eq(tex.sym('Σ′'), tex.op('='), tex.sym('D'), Sig, tex.sym('D'), tex.op(','), tex.sym('D'), tex.op('='), tex.text('diag(1, 1, ' + LA.fmt(Math.round(Math.pow(100, s.scale3))) + ')')));
          P.push(tex.eq(tex.text('PC1'), tex.op('='), tex.vec(f.q[0], { color: 'gold' }), s.stdT > 0.5 ? tex.text('standardized', { color: 'ok' }) : tex.text(s.scale3 > 0.9 ? '≈ e₃: units, not structure' : '', { color: 'bad' })));
          break; }
        case 'svd':
          P.push(tex.eq(Xc, tex.op('='), tex.sym('U'), Sig, tex.T(tex.sym('V')), tex.op(','), tex.sym('V'), tex.op('='), tex.sym('Q')));
          P.push(tex.eq(tex.sym('σ', { sub: 'i', sup: '2' }), tex.op('/'), tex.sym('n'), tex.op('='), tex.sym('λ', { sub: 'i' }), tex.op(':'), tn(s.svd.s2n[0], 'gold'), tex.op(','), tn(s.svd.s2n[1], 'gold'), tex.op(','), tn(s.svd.s2n[2], 'gold')));
          break;
        case 'digits':
          P.push(tex.eq(tex.sym('X'), tex.op('∈'), tex.text('ℝ⁹⁰ˣ⁶⁴'), tex.op(','), tex.sym('Z'), tex.op('='), Xc, tex.sym('V', { sub: '2' })));
          P.push(tex.eq(tex.text('explained'), tex.op('='), pct(s.dig.expl[0]), tex.op('+'), pct(s.dig.expl[1]), tex.op('='), pct(s.dig.cum2)));
          break;
        case 'ml': case 'recap':
          if (s.panel === 'recap') P.push(tex.eq(Sig, tex.sym('q', { sub: 'i', color: 'gold' }), tex.op('='), tex.sym('λ', { sub: 'i' }), tex.sym('q', { sub: 'i', color: 'gold' })));
          P.push(tex.eq(tex.sym('Z'), tex.op('='), Xc, tex.sym('Q', { sub: 'k' })));
          break;
        default: { // play
          if (s.view === '3d') {
            const f = frame3(covN(base3(s))), tot = f.lam[0] + f.lam[1] + f.lam[2];
            P.push(tex.eq(tex.text('PC1'), tex.op('='), tex.vec(f.q[0], { color: 'gold' }), tex.text('k = ' + Math.round(s.k))));
            P.push(tex.eq(tex.text('explained'), tex.op('='), pct(f.lam[0] / tot), tex.op(','), pct(f.lam[1] / tot), tex.op(','), pct(f.lam[2] / tot)));
          } else {
            const B = base2(s), S = covN(B), e = N.eigSym(S), dd = [Math.cos(s.theta), Math.sin(s.theta)];
            P.push(tex.eq(tex.text('var(shadows)'), tex.op('='), tex.T(d), Sig, d, tex.op('='), tn(N.dot(dd, N.matVec(S, dd)), 'mag'), tex.op(','), tex.text('max'), tex.op('='), tn(e.values[0], 'gold')));
            P.push(tex.eq(tex.text('PC1 explains'), pct(e.values[0] / (e.values[0] + e.values[1]))));
          }
        }
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Which line captures the most spread?', duration: 5, interactive: true,
        narration: 'The centered cloud from last lesson. Lay a line through the origin and drop every point’s shadow onto it (magenta). Some lines collect a wide scatter of shadows; others squeeze them together. Turn the line: the variance of the shadows rises and falls, and peaks at the gold direction — the first <b>principal component</b>. Drag the white handle yourself.',
        setup(s) { reset2(s, { panel: 'var', showLine: true, lineHandle: true, theta: 0, sweep: 0, bars: 'var' }); },
        at(s, t) {
          const th1 = pc1Angle(s), u = seg(t, 0, 0.7, 'linear');
          s.sweep = u;
          s.theta = t < 0.7 ? Math.PI * u : LA.lerp(Math.PI, Math.PI + th1, seg(t, 0.72, 0.95));
          if (s.theta > Math.PI + 1e-9) s.theta -= Math.PI;
          s.pcT = seg(t, 0.85, 1);
        } },

      { title: 'It is the covariance’s top eigenvector', duration: 4,
        narration: 'No need to search: the variance of shadows along a unit direction d is dᵀΣd, a quadratic form, and the direction that maximizes it is Σ’s top eigenvector ' + goldS('q₁') + ' — p4-06’s long ellipse axis. The second component ' + goldS('q₂') + ' is perpendicular. Their eigenvalues are the variances along them: 2.8 and 0.2.',
        setup(s) { reset2(s, { panel: 'eig', showLine: true, sweep: 1, pcT: 1, bars: 'lam' }); s.theta = pc1Angle(s); },
        at(s, t) { s.ellT = seg(t, 0.1, 0.7); } },

      { title: 'Rotate into the gold frame', duration: 4,
        narration: 'Change basis (p4-01) to the components: rotate the whole cloud so ' + goldS('q₁') + ' lies along the horizontal axis. Now the two new coordinates are uncorrelated — the covariance turns diagonal — and the first coordinate alone holds 93 % of the spread. That share λ₁ / (λ₁ + λ₂) is its <b>explained variance</b>.',
        setup(s) { reset2(s, { panel: 'rot', pcT: 1, ellT: 1, tileT: 1, bars: 'pct' }); },
        at(s, t) { s.rotT = seg(t, 0.1, 0.85); } },

      { title: 'Flatten: keep one number per point', duration: 4,
        narration: 'Drop the second coordinate. Every point slides onto the gold line; the dashed segments are what was thrown away — short, because the cloud was thin that way. Eighty pairs of numbers became eighty single numbers: the <b>projection onto PCs</b>, keeping 93 % of the variance.',
        setup(s) { reset2(s, { panel: 'flat', pcT: 1, ellT: 0.0, rotT: 1, bars: 'pct' }); },
        at(s, t) { s.flatT = seg(t, 0.1, 0.8); } },

      { title: 'A 3D cloud', duration: 4.5,
        narration: 'Now three features: 150 synthetic points in a long, flat, tilted cloud — nearly a pancake floating at an angle. Its covariance ellipsoid has three gold axes with spreads 3, 1.2 and 0.3. The third axis is tiny: the cloud is almost two-dimensional, but it lies in a tilted plane that no feature axis sees.',
        setup(s) { reset3(s, { panel: '3d', n3: 0, ellT: 1 }); },
        at(s, t) { s.n3 = Math.round(150 * seg(t, 0, 0.5, 'linear')); LA.cam.tween(s, { yaw: 1.6 }, { yaw: 2.0 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Rotate, then crop', duration: 6,
        narration: 'Rotate the cloud until the gold axes line up with the x, y, z axes — the change of basis. Then collapse the shortest axis: points slide along ' + goldS('q₃') + ' onto the gold plane of ' + goldS('q₁') + ', ' + goldS('q₂') + '. The ghosts show what moved; the bars show what was kept: 85 %, 14 %, 1 %. PCA is a rotation followed by a crop.',
        setup(s) { reset3(s, { panel: 'crop', bars: 'pct3' }); },
        at(s, t) { s.rotT = seg(t, 0.05, 0.45); s.planeT = seg(t, 0.5, 0.6); s.flatT = seg(t, 0.55, 0.95); } },

      { title: 'Reconstruction: back in the original room', duration: 5,
        narration: 'Rotate back with Q. The flattened points land near their originals; the gaps are the <b>reconstruction</b> errors — all along ' + goldS('q₃') + ', all small. Compressing 150×3 numbers to 150×2 cost 1 % of the spread. Keep only one axis and the points fall onto the gold line: the price jumps to 15 %.',
        setup(s) { reset3(s, { panel: 'recon', bars: 'pct3', rotT: 1, flatT: 1, planeT: 1 }); },
        at(s, t) { s.rotT = 1 - seg(t, 0.05, 0.45); s.kT = seg(t, 0.62, 0.92); s.planeT = 1 - seg(t, 0.6, 0.75); } },

      { title: 'Misconception: PCA finds “important” features', duration: 5, misconception: true,
        narration: '✕ “PC1 is the most important direction in the data.” Multiply feature 3 by 100 — meters to centimeters — and PC1 swings almost straight along feature 3, because PCA chases <i>variance</i>, whatever its units. Standardize the features first, or PCA measures your rulers, not your data.',
        setup(s) { reset3(s, { panel: 'units', ellT: 0, ghosts: false }); },
        at(s, t) {
          s.scale3 = seg(t, 0.05, 0.4); s.cross = t > 0.42 && t < 0.62; s.stdT = seg(t, 0.62, 0.85); s.okMark = t > 0.88;
          const dd = camDistFor(s); LA.cam.tween(s, { dist: dd }, { dist: dd }, 1);
        } },

      { title: 'PCA is the SVD of the centered data', duration: 4,
        narration: 'Take the SVD of the 150×3 centered matrix: Xc = UΣVᵀ. Its right singular vectors V are the gold components, and σᵢ²/n equals λᵢ exactly. Two routes, one answer — <b>PCA = SVD</b> — and the SVD route is how libraries do it, because it never forms the 784×784 covariance.',
        setup(s) { reset2(s, { panel: 'svd', scene: 'svd', svdT: 0 }); s.cam = cam2(0, 0, 3.6); },
        at(s, t) { s.svdT = seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Ninety digits in 64 dimensions: a shadow', duration: 5.5,
        narration: 'Ninety synthetic 8×8 digit-like images of three kinds. Each is a point in ℝ⁶⁴ — not drawable. PCA finds the two directions of greatest spread and we plot the shadows: the kinds separate into clusters. The label says it plainly: a shadow of ℝ⁶⁴, keeping the stated share of variance. The picture stops at 3. The formula does not. Hover a point.',
        setup(s) { reset2(s, { panel: 'digits', scene: 'digits', digT: 0 }); s.cam = cam2(0, 0, 3.6); },
        at(s, t) { s.digT = seg(t, 0, 0.95, 'linear'); } },

      { title: 'Why ML cares', duration: 4, mlTie: true,
        narration: 'PCA is how embeddings get visualized (always a shadow — read the %), how features are decorrelated before regression, how images are compressed and denoised, and how researchers probe what a layer encodes. Whitening, next, divides each component by √λ to make the cloud round.',
        setup(s) { reset2(s, { panel: 'ml', scene: 'ml', mlT: 0 }); s.cam = cam2(0, 0, 3.6); },
        at(s, t) { s.mlT = seg(t, 0.1, 0.85); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Center. Σ’s gold eigenvectors are the directions of spread; eigenvalues are the variances along them. Rotate into that frame, crop the small axes, keep the % the bars show. Standardize first or units win. Shadows of high-dimensional clouds are honest only with their % attached.',
        cards: ['<b>Line of greatest spread.</b> The variance of shadows dᵀΣd peaks at Σ’s top eigenvector q₁.', '<b>Rotate, then crop.</b> Keep the top k components; the bars (85 / 14 / 1) say what survives.', '<b>Shadows carry a % label.</b> A 2-D picture of ℝ⁶⁴ is honest only with its explained variance.'],
        setup(s) { reset2(s, { panel: 'recap', scene: 'recap' }); s.cam = cam2(0, 0, 3.6); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Turn the candidate line in 2D and watch the shadows’ variance; switch to the 3D pancake and choose how many components to keep (with ghosts showing what is lost); stretch feature 3 by 100 to see PCA chase units. The Digits preset shows the 64-D shadow; hover a point.</p>',
      setup(s) { reset2(s, { panel: 'play', showLine: true, lineHandle: true, sweep: 1, pcT: 1, ellT: 1, bars: 'var', useK: true, k: 2 }); s.theta = 1.2; },
      controls: [
        { type: 'slider', bind: 'theta', min: 0, max: 3.1416, step: 0.005, label: 'candidate line (2D)', color: 'mag', fmt: x => LA.fmt.deg(x) },
        { type: 'slider', bind: 'k', min: 1, max: 3, step: 1, label: 'components kept k', color: 'gold', fmt: x => String(Math.round(x)) },
        { type: 'toggle', bind: 'ghosts', label: 'ghost points (what is lost)' },
        { type: 'slider', bind: 'scale3', min: 0, max: 1, step: 0.01, label: 'feature 3 × (1 … 100)', color: 'bad', fmt: x => '×' + LA.fmt(Math.round(Math.pow(100, x))) },
        { type: 'preset', label: 'Data', items: [
          { label: '2D cloud', apply(s) { reset2(s, { panel: 'play', showLine: true, lineHandle: true, sweep: 1, pcT: 1, ellT: 1, bars: 'var', useK: true, k: 2, theta: s.theta }); } },
          { label: '3D pancake', apply(s) { reset3(s, { panel: 'play', useK: true, k: 2, ellT: 1, cloud3: 'pancake', bars: 'pct3', scale3: s.scale3 }); } },
          { label: 'Round 3D (equal λ — components ambiguous)', apply(s) { reset3(s, { panel: 'play', useK: true, k: 2, ellT: 1, cloud3: 'round', bars: 'pct3' }); } },
          { label: 'Digits (64-D shadow)', apply(s) { reset2(s, { panel: 'digits', scene: 'digits', digT: 1 }); s.cam = cam2(0, 0, 3.6); } },
        ] },
        { type: 'info', html: '<b>Tip.</b> In 3D, left-drag orbits and the wheel zooms. With k = 2 the dashed segments run along q₃ only.' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Turn the candidate line (drag its white handle) to the direction of greatest spread.',
        setup(s) { reset2(s, { panel: 'var', showLine: true, lineHandle: true, sweep: 0, bars: 'var' }); s.theta = Math.PI / 2; },
        judge(s) { const th1 = pc1Angle(s); const dAng = Math.abs(Math.atan2(Math.sin(2 * (s.theta - th1)), Math.cos(2 * (s.theta - th1)))) / 2 * 180 / Math.PI; return { ok: dAng < LA.TOL.angleDeg, progress: Math.max(0, 1 - dAng / 90) }; },
        hint: 'Watch the variance bar — stop at its peak.',
        solution(s) { s.theta = pc1Angle(s); } },
      { kind: 'manipulate',
        prompt: 'Reshape the cloud (drag the white handles: spread x, spread y and the ρ track) so that PC1 explains at least 95 % of the variance.',
        setup(s) { reset2(s, { panel: 'rot', src: 'gen', sx: 1.5, sy: 0.8, rho: 0.6, pcT: 1, ellT: 1, bars: 'pct', genHandles: true }); s.cam = cam2(0, -0.6, 4.0); },
        judge(s) { const e = N.eigSym(genSigma(s.sx, s.sy, s.rho)); const f = e.values[0] / (e.values[0] + e.values[1]); return { ok: f >= 0.95, progress: LA.clamp((f - 0.5) / 0.45, 0, 1) }; },
        hint: 'Make the cloud long and thin: a high correlation or very unequal spreads.',
        solution(s) { s.rho = 0.95; } },
      { kind: 'manipulate',
        prompt: 'The 3D cloud is rotated into its gold frame. Orbit until you look straight down the third component ' + goldS('q₃') + ' (the pancake then faces you and looks widest).',
        setup(s) { reset3(s, { panel: 'crop', rotT: 1, ellT: 1, bars: 'pct3' }); },
        judge(s) { const f = frame3(covN(base3(s))); const q3 = N.matVec(rotTo(N.transpose(N.fromCols(f.q)), s.rotT), f.q[2]); return LA.judge.lookAlong(s.cam, q3, 10); },
        hint: 'You want the pancake face-on: look from straight above.',
        solution(s) { s.cam = LA.cam.lookAlong(s.cam, 'z'); } },
    ],
  });

  /* ---------- state helpers ---------- */
  function cam2(x, y, h) { return Object.assign(LA.cam.defaults2d(), { center: [x, y], halfH: h }); }
  function cam3() { return Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0], dist: 16, yaw: 2.0, pitch: 0.4 }); }
  function reset2(s, o) { Object.assign(s, DEF, { view: '2d' }, o || {}); s.cam = cam2(0, 0, 4.0); if (o && o.scene && o.scene !== 'two') s.cam = cam2(0, 0, 3.6); }
  function reset3(s, o) { Object.assign(s, DEF, { view: '2d', scene: 'three' }, o || {}); s.view = '3d'; s.cam = cam3(); }
  function base2(s) { if (s.src === 'gen') { const L = chol2(s.sx, s.sy, s.rho); return s.z49.map(z => N.matVec(L, z)); } return s.pts2; }
  function pc1Angle(s) { const q = N.eigSym(covN(base2(s))).vectors[0]; let a = Math.atan2(q[1], q[0]); if (a < 0) a += Math.PI; if (a >= Math.PI) a -= Math.PI; return a; }
  function sdOf(P) { const S = covN(P); return [0, 1, 2].map(i => Math.sqrt(Math.max(S[i][i], 1e-12))); }
  function base3(s) {
    let P = s.p3;
    if (s.cloud3 === 'round') P = P.map(p => N.scale(N.matVec(L3inv, p), 1.5));
    if (s.scale3 > 0 || s.stdT > 0) {
      const sc = Math.pow(100, s.scale3), sd = sdOf(P);
      P = P.map(p => LA.lerpV([p[0], p[1], p[2] * sc], [p[0] / sd[0], p[1] / sd[1], p[2] / sd[2]], s.stdT));
    }
    return P;
  }
  function camDistFor(s) { const sd = sdOf(s.p3)[2], sc = Math.pow(100, s.scale3); const scaled = 14 * Math.max(1, 2.6 * sd * sc / 4.5); return LA.lerp(scaled, 10, s.stdT); }

  /* ---------- 2D scene ---------- */
  function render2(g, s, ctx) {
    const B = base2(s), S = covN(B), e = N.eigSym(S), H = sqrtSym(S);
    const q1 = e.vectors[0], q2 = e.vectors[1], th1 = Math.atan2(q1[1], q1[0]);
    const R = N.rot2(-th1 * s.rotT), q1r = N.matVec(R, q1), q2r = N.matVec(R, q2);
    const f = s.useK ? (s.k < 1.5 ? 1 : 0) : s.flatT;
    const G = B.map(p => N.matVec(R, p));
    const P = G.map(p => N.sub(p, N.scale(q2r, N.dot(p, q2r) * f)));
    g.ghostGrid(); g.axes();
    if (s.ellT > 0) g.ellipse(lerpM(I2(), N.matMul(R, H), s.ellT), [0, 0], { stroke: 'teal', width: 2 });
    // polar plot of var(θ): r = dᵀΣd
    if (s.sweep > 0 && s.rotT < 0.01) {
      const pts = [], n = Math.max(2, Math.round(90 * s.sweep));
      for (let k = 0; k <= n; k++) { const a = Math.PI * s.sweep * k / n, d = [Math.cos(a), Math.sin(a)]; pts.push(N.scale(d, N.dot(d, N.matVec(S, d)))); }
      g.polyline(pts, { color: 'w', width: 1.6, alpha: 0.7 });
      g.polyline(pts.map(p => [-p[0], -p[1]]), { color: 'w', width: 1.6, alpha: 0.7 });
      if (s.pcT > 0.5) g.marker.ring(N.scale(q1, e.values[0]), { r: 9, color: 'gold' });
      if (s.sweep > 0.6) g.label('var(d) in every direction', N.scale(q1, -e.values[0]), { color: 'w', size: 12, font: 'ui', dy: 18, anchor: 'left', dx: -30 });
    }
    // ghosts and residuals of the flattening
    if (f > 0 && s.ghosts) {
      g.points(G, { r: 3, color: 'muted', alpha: 0.35 });
      P.forEach((p, i) => g.line(G[i], p, { color: 'residual', dashed: [3, 3], width: 1, alpha: 0.8 }));
    }
    // the candidate line, drops and magenta shadows
    if (s.showLine && s.rotT < 0.01) {
      const d = [Math.cos(s.theta), Math.sin(s.theta)];
      g.lineThrough([0, 0], d, { color: 'hud', width: 1.6, len: 7, alpha: 0.85 });
      B.forEach(p => g.line(p, N.scale(d, N.dot(p, d)), { color: 'residual', dashed: [3, 4], width: 1, alpha: 0.4 }));
      g.points(B.map(p => N.scale(d, N.dot(p, d))), { r: 3, color: 'mag', alpha: 0.9 });
    }
    // gold principal axes
    if (s.pcT > 0) {
      g.lineThrough([0, 0], q1r, { color: 'gold', width: 2, alpha: 0.45 * s.pcT, len: 7 });
      g.vector(N.scale(q1r, Math.sqrt(e.values[0])), { color: 'gold', width: 3, head: 11, alpha: s.pcT, label: 'q₁', labelFont: 'math' });
      if (s.ellT > 0.5 || s.rotT > 0 || s.useK) g.vector(N.scale(q2r, Math.sqrt(Math.max(e.values[1], 0))), { color: 'gold', width: 3, head: 9, alpha: s.pcT, label: 'q₂', labelFont: 'math' });
    }
    g.points(P, { r: 3.6, color: 'data' });
    if (s.showLine && s.rotT < 0.01) {
      const d = [Math.cos(s.theta), Math.sin(s.theta)];
      const hp = N.scale(d, 3.0);
      g.label('d', hp, { color: 'hud', dx: 14, dy: -12, font: 'math' });
      if (s.lineHandle) g.handle('line', hp, (p, st) => { let a = Math.atan2(p[1], p[0]); if (a < 0) a += Math.PI; if (a >= Math.PI) a -= Math.PI; st.theta = a; }, { color: 'white' });
    }
    // HUD bars
    const x = 22, y = 84, h = 130;
    if (s.bars === 'var' && s.showLine) {
      const d = [Math.cos(s.theta), Math.sin(s.theta)], v = N.dot(d, N.matVec(S, d));
      g.bars([v], { screen: true, x, y, w: 44, h, max: 3, colors: [Math.abs(v - e.values[0]) < 0.03 ? 'gold' : 'mag'], title: 'variance of shadows' });
      g.hud('now ' + LA.fmt.fixed(v, 2), x, y + h + 16, { size: 13, color: 'mag' });
      g.label('◂ max λ₁ = ' + LA.fmt.fixed(e.values[0], 2), [x + 48, y + h - e.values[0] / 3 * h], { screen: true, anchor: 'left', size: 11.5, color: 'gold', font: 'ui' });
    }
    if (s.bars === 'lam') g.bars([e.values[0], e.values[1]], { screen: true, x, y, w: 90, h, max: 3, colors: ['gold', 'gold'], labels: ['λ₁', 'λ₂'], values: true, title: 'variance along q₁, q₂' });
    if (s.bars === 'pct') {
      const tot = e.values[0] + e.values[1], kept = [e.values[0] / tot, e.values[1] / tot];
      g.bars(kept.map(v => 100 * v), { screen: true, x, y, w: 90, h, max: 115, colors: ['gold', f > 0.5 || s.genHandles ? 'muted' : 'gold'], labels: ['PC1', 'PC2'], title: 'explained variance (%)' });
      kept.forEach((v, i) => g.hud(Math.round(100 * v) + ' %', x + 4 + i * 47, y + h - v * h / 1.15 - 10, { size: 12.5, color: i === 0 ? 'gold' : 'hud' }));
    }
    if (s.tileT > 0) tile(g, covN(P), { name: 'Σ′ =' });
    if (s.genHandles) drawGenHandles(g, s);
  }

  function drawGenHandles(g, s) {
    const ty = -3.3;
    const hx = [s.sx, 0], hy = [0, s.sy], hr = [2 * s.rho, ty];
    g.line([-2, ty], [2, ty], { color: 'muted', width: 2, alpha: 0.7 });
    [-2, 0, 2].forEach(k => g.line([k, ty - 0.08], [k, ty + 0.08], { color: 'muted', width: 1.5 }));
    g.label('ρ = ' + LA.fmt.fixed(s.rho, 2), [2.15, ty], { anchor: 'left', color: 'b', size: 13, font: 'ui' });
    g.point(hx, { r: 4, color: 'i' }); g.label('σx', hx, { color: 'i', size: 13, dy: 16, font: 'math' });
    g.point(hy, { r: 4, color: 'j' }); g.label('σy', hy, { color: 'j', size: 13, dx: -18, font: 'math' });
    g.point(hr, { r: 4, color: 'b' });
    g.handle('sx', hx, (p, st) => { st.sx = LA.clamp(Math.abs(p[0]), 0.2, 3); }, { color: 'white' });
    g.handle('sy', hy, (p, st) => { st.sy = LA.clamp(Math.abs(p[1]), 0.2, 3); }, { color: 'white' });
    g.handle('rho', hr, (p, st) => { st.rho = LA.clamp(p[0] / 2, -0.99, 0.99); }, { color: 'white' });
  }

  /* ---------- 3D scene ---------- */
  function render3(g, s) {
    const B = base3(s), fr = frame3(covN(B)), q = fr.q, lam = fr.lam;
    const Q = N.fromCols(q), Qt = N.transpose(Q), F = rotTo(Qt, s.rotT), FQ = N.matMul(F, Q);
    const f3 = s.useK ? (s.k < 2.5 ? 1 : 0) : s.flatT, f2 = s.useK ? (s.k < 1.5 ? 1 : 0) : s.kT;
    const n = Math.min(B.length, Math.round(s.n3));
    const big = s.scale3 > 0.05 && s.stdT < 0.95;
    if (!big) { g.grid(null, { range: 6, alpha: 0.28 }); g.axes({ length: 5.5 }); }
    else g.axes({ length: Math.max(5.5, camDistFor(s) * 0.3), names: ['x', 'y', 'z (feature 3)'] });
    const G = [], P = [];
    for (let i = 0; i < n; i++) {
      const c = N.matVec(Qt, B[i]);
      G.push(N.matVec(F, B[i]));
      P.push(N.matVec(FQ, [c[0], c[1] * (1 - f2), c[2] * (1 - f3)]));
    }
    if (s.planeT > 0 || (s.useK && s.k < 2.5 && s.k > 1.5)) g.plane({ point: [0, 0, 0], span: [N.matVec(F, q[0]), N.matVec(F, q[1])], extentU: 6.5, extentV: 3, color: 'gold', alpha: 0.16 * (s.useK ? 1 : s.planeT) });
    if (s.ellT > 0 && !big) g.ellipsoid(N.matMul(FQ, N.diag(lam.map(Math.sqrt))), { color: 'teal', meridians: 4, alpha: 0.75 });
    const moved = (f3 > 0 || f2 > 0);
    if (moved && s.ghosts) for (let i = 0; i < n; i++) { g.point(G[i], { r: 2.4, color: 'muted', alpha: 0.4 }); if (f3 > 0.02 || f2 > 0.02) g.line(G[i], P[i], { color: 'residual', dashed: [3, 3], width: 1, alpha: 0.75 }); }
    g.points(P, { r: 2.8, color: 'data', alpha: 0.9 });
    if (s.axesT > 0) q.forEach((qi, i) => {
      const L = Math.sqrt(lam[i]); if (L < 1e-3) return;
      const tip = N.matVec(F, N.scale(qi, L));
      g.vector([0, 0, 0], tip, { color: 'gold', width: 3.2, head: L < 0.6 ? 7 : 12, label: 'q' + SUB[i + 1], labelFont: 'math' });
    });
    if (s.panel === 'units') {
      const tip = N.matVec(F, N.scale(q[0], Math.sqrt(lam[0])));
      if (s.cross) { g.marker.cross(tip, { dx: 26, dy: 4, size: 22 }); g.label('units, not structure', tip, { color: 'bad', dx: 90, dy: 28, size: 13.5, font: 'ui' }); }
      if (s.okMark) { g.marker.check(tip, { dx: 24, dy: -4, size: 22 }); g.hud('PC1 is now a real direction of spread ✓', 16, 46, { size: 13, color: 'ok' }); }
      g.hud(s.stdT > 0.5 ? 'each feature ÷ its own spread' : 'feature 3 × ' + LA.fmt(Math.round(Math.pow(100, s.scale3))) + '  (meters → centimeters)', 16, 24, { size: 14, color: s.stdT > 0.5 ? 'ok' : 'bad', bold: true });
    } else {
      g.hud('150 synthetic points (LA.rng seed 50)' + (s.cloud3 === 'round' ? ' · round: λ₁ = λ₂ = λ₃, components arbitrary' : ''), 16, 24, { size: 13, color: 'muted' });
      q.forEach((qi, i) => g.hud('q' + SUB[i + 1] + ':  √λ' + SUB[i + 1] + ' = ' + LA.fmt.fixed(Math.sqrt(lam[i]), 2), 16, 48 + 19 * i, { size: 13, color: 'gold' }));
    }
    if (s.bars === 'pct3') {
      const tot = lam[0] + lam[1] + lam[2], kept = [true, f2 < 0.5, f3 < 0.5];
      const x = g.size.w - 170, y = 40, h = 110;
      g.bars(lam.map(l => 100 * l / tot), { screen: true, x, y, w: 140, h, max: 100, colors: kept.map(k => k ? 'gold' : 'muted'), labels: ['PC1', 'PC2', 'PC3'], title: 'explained variance (%)' });
      lam.forEach((l, i) => g.hud(Math.round(100 * l / tot) + ' %', x + 6 + i * 48, y + h - (l / tot) * h - 10, { size: 12.5, color: kept[i] ? 'gold' : 'hud' }));
      const lost = (f3 > 0.5 ? lam[2] : 0) + (f2 > 0.5 ? lam[1] : 0);
      if (lost > 0) g.hud('lost: ' + Math.round(100 * lost / tot) + ' %', x, y + h + 30, { size: 13, color: 'hud' });
    }
  }

  /* ---------- PCA = SVD: two columns of numbers ---------- */
  function renderSVD(g, s) {
    const W = g.size.w, sv = s.svd, u = s.svdT;
    const xL = W * 0.06, xR = W * 0.54, y0 = 70;
    const vec = v => '[' + v.map(x => LA.fmt.fixed(x, 2)).join(', ') + ']';
    g.hud('eigSym(Σ₃): eigenvectors, eigenvalues', xL, y0 - 30, { size: 13, color: 'muted' });
    g.hud('svd(Xc): right singular vectors, σ²/n', xR, y0 - 30, { size: 13, color: 'muted' });
    for (let i = 0; i < 3; i++) {
      const y = y0 + i * 62, show = u > 0.15 + 0.25 * i, ring = u > 0.25 + 0.25 * i;
      g.hud('q' + SUB[i + 1] + ' = ' + vec(sv.q[i]), xL, y, { size: 14, color: 'gold', font: 'mono' });
      g.hud('λ' + SUB[i + 1] + ' = ' + LA.fmt.fixed(sv.lam[i], 2), xL, y + 22, { size: 14, color: 'gold', font: 'mono' });
      if (show) {
        g.hud('v' + SUB[i + 1] + ' = ' + vec(sv.v[i]), xR, y, { size: 14, color: 'gold', font: 'mono' });
        g.hud('σ' + SUB[i + 1] + '²/n = ' + LA.fmt.fixed(sv.s2n[i], 2), xR, y + 22, { size: 14, color: 'gold', font: 'mono' });
      }
      if (ring) { g.label('=', [W * 0.5, y + 11], { screen: true, color: 'gold', size: 20, font: 'ui', bold: true }); g.marker.ring(g.P.toWorld(W * 0.5, y + 11), { r: 14, color: 'gold' }); }
    }
    // shape tile: Xc (150×3) = U (150×3) · Σ (3×3) · Vᵀ (3×3)
    const by = -0.2, hT = 2.0, wT = 0.55, sq = 0.55;
    const rect = (x0, y1, w, h, col, lab, sub) => { g.polygon([[x0, y1 - h], [x0 + w, y1 - h], [x0 + w, y1], [x0, y1]], { fill: col, alpha: 0.25, stroke: col, width: 1.5 }); g.label(lab, [x0 + w / 2, y1 - h / 2], { color: 'text', size: 15, font: 'math' }); g.label(sub, [x0 + w / 2, y1 - h], { anchor: 'below', dy: 6, color: 'muted', size: 11.5, font: 'ui' }); };
    const a = seg(u, 0.75, 1);
    rect(-2.6, by, wT, hT, 'data', 'Xc', '150×3');
    g.label('=', [-1.75, by - hT / 2], { size: 22, color: 'hud', font: 'ui' });
    if (a > 0) {
      rect(-1.35, by, wT, hT, 'teal', 'U', '150×3');
      rect(-0.45, by, sq, sq, 'gold', 'Σ', '3×3');
      rect(0.45, by, sq, sq, 'gold', 'Vᵀ', '3×3');
      g.label('rows of Vᵀ = the gold components', [0.72, by - sq - 0.65], { anchor: 'left', color: 'gold', size: 12.5, font: 'ui' });
    }
    g.hud('Two routes, one answer: V = Q and σᵢ²/n = λᵢ', 16, g.size.h - 22, { size: 14, color: 'hud', bold: true });
  }

  /* ---------- 64-D digits: an honest shadow ---------- */
  function digitScale(s) { let m = 0; s.dig.Z.forEach(z => { m = Math.max(m, Math.abs(z[0]), Math.abs(z[1])); }); return 2.1 / (m || 1); }
  function renderDigits(g, s, ctx) {
    const D = digitsData(), dg = s.dig, u = s.digT, k = digitScale(s), c0 = [-0.55, 0.05];
    const W = g.size.w;
    // phase 1–2: thumbnails, then their 64-cell strips (the image "unrolled")
    const a1 = 1 - seg(u, 0.5, 0.62);
    if (a1 > 0.01) {
      for (let r = 0; r < 6; r++) {
        const i = r * 15 + 1, y1 = 2.9 - r * 0.95;
        g.image(toGrid(D.X[i]), { x0: -2.9, x1: -2.1, y0: y1 - 0.8, y1 }, { alpha: a1 });
        if (u > 0.18) {
          const sp = g.P.toScreen([-1.9, y1 - 0.4]);
          const w = (W - sp[0] - 14) * seg(u, 0.18, 0.4);
          if (w > 4) g.strip(D.X[i], { x: sp[0], y: sp[1] - 9, w, h: 18, range: [0, 1] });
        }
      }
      if (u > 0.25) g.hud('each image = 64 numbers = one point in ℝ⁶⁴', 16, 24, { size: 13.5, color: 'hud', alpha: a1 });
    }
    // phase 3: the 2-D shadow
    const a3 = seg(u, 0.55, 0.85);
    if (a3 > 0) {
      const pts = dg.Z.map(z => [c0[0] + k * z[0], c0[1] + k * z[1]]);
      const pa = seg(u, 0.8, 1);
      if (pa > 0) {
        g.vector([c0[0] - 2.3, c0[1]], [c0[0] + 2.25, c0[1]], { color: 'gold', width: 2, head: 9, alpha: pa, label: 'PC1', labelFont: 'ui', dx: -4, dy: 16 });
        g.vector([c0[0], c0[1] - 2.3], [c0[0], c0[1] + 2.45], { color: 'gold', width: 2, head: 9, alpha: pa, label: 'PC2', labelFont: 'ui' });
      }
      const hot = ctx.hot && /^d\d+$/.test(ctx.hot) ? +ctx.hot.slice(1) : null;
      g.points(pts, { r: 4, colors: i => CLASS[dg.lab[i]], alpha: 0.92 * a3 });
      pts.forEach((p, i) => g.handle('d' + i, p, () => { }, { show: false, always: true, r: 7, onClick(st) { st.hotDigit = i; } }));   // hover shows a digit; a tap (touch) pins it
      g.hud('shadow of ℝ⁶⁴ on 2 directions — ' + Math.round(100 * dg.cum2) + ' % of variance', 16, 24, { size: 14, color: 'gold', bold: true, alpha: a3 });
      g.hud('PC1 ' + Math.round(100 * dg.expl[0]) + ' % + PC2 ' + Math.round(100 * dg.expl[1]) + ' %; the other ' + Math.round(100 * (1 - dg.cum2)) + ' % is invisible here', 16, 44, { size: 12.5, color: 'muted', alpha: a3 });
      ['kind 0', 'kind 1', 'kind 2'].forEach((t, j) => { g.point([-2.75, -2.55 - 0.3 * j], { r: 4.5, color: CLASS[j], alpha: a3 }); g.label('synthetic “' + j + '”', [-2.6, -2.55 - 0.3 * j], { anchor: 'left', color: CLASS[j], size: 12, font: 'ui', alpha: a3 }); });
      // the hovered (or default) digit: original and its 2-number reconstruction
      if (a3 > 0.9) {
        const i = hot === null ? s.hotDigit : hot;
        const z = dg.Z[i];
        let rec = dg.mean.slice(); for (let j = 0; j < 2; j++) rec = N.add(rec, N.scale(dg.comps[j], z[j]));
        g.marker.ring(pts[i], { r: 9, color: 'white' });
        g.image(toGrid(D.X[i]), { x0: 2.0, x1: 2.85, y0: 2.0, y1: 2.85 });
        g.image(toGrid(rec), { x0: 2.0, x1: 2.85, y0: 0.75, y1: 1.6 });
        g.label('this point', [2.42, 2.0], { anchor: 'below', dy: 4, color: 'hud', size: 11.5, font: 'ui' });
        g.label('from 2 numbers', [2.42, 0.75], { anchor: 'below', dy: 4, color: 'hud', size: 11.5, font: 'ui' });
        g.label(hot === null ? 'hover or tap a point' : 'hovered', [2.42, 2.85], { anchor: 'above', dy: -4, color: 'muted', size: 11.5, font: 'ui' });
      }
    }
    g.hud(HONEST, 16, g.size.h - 22, { size: 14, color: 'hud', bold: true });
  }

  /* ---------- ML tie-in rows ---------- */
  function renderML(g, s) {
    const D = digitsData(), dg = s.dig, u = s.mlT, k = digitScale(s) * 0.85;
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-2.6, 0, 2.3) }, (g2, r) => {
      g2.points(dg.Z.map(z => [k * z[0], k * z[1]]), { r: 3, colors: i => CLASS[dg.lab[i]], alpha: 0.4 + 0.6 * u });
      g2.hud('embedding scatter = a shadow', r.x + 12, r.y + 22, { size: 13.5, color: 'gold', bold: true });
      g2.hud('2 of 64 dims, ' + Math.round(100 * dg.cum2) + ' % of variance', r.x + 12, r.y + 42, { size: 12.5, color: 'hud' });
      g2.hud('(a 768-dim embedding: read its %)', r.x + 12, r.y + 60, { size: 12, color: 'muted' });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(0, 0, 2.3) }, (g2, r) => {
      const ims = [D.X[dg.idx], dg.rec[0], dg.rec[1], dg.rec[2]], labs = ['original', 'k = 2 (' + Math.round(100 * dg.cum2) + ' %)', 'k = 8 (' + Math.round(100 * dg.cum8) + ' %)', 'k = 32 (' + Math.round(100 * dg.cum32) + ' %)'];
      ims.forEach((im, j) => { const x0 = -3.9 + j * 2.15, al = j === 0 ? 1 : seg(u, 0.1 + 0.25 * (j - 1), 0.35 + 0.25 * (j - 1)); g2.image(toGrid(im), { x0, x1: x0 + 1.5, y0: -1.0, y1: 0.5 }, { alpha: al }); g2.label(labs[j], [x0 + 0.75, -1.0], { anchor: 'below', dy: 6, color: j === 0 ? 'hud' : 'gold', size: 12, font: 'ui', alpha: al }); });
      g2.hud('compression and denoising: rebuild from k numbers', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-2.2, 0, 2.3) }, (g2, r) => {
      const e = N.eigSym([[2, 1.2], [1.2, 1]]), Q = N.fromCols(e.vectors);
      const M = N.matMul(Q, N.diag(e.values.map(l => LA.lerp(Math.sqrt(l), 1, u))));
      g2.ellipse(M, [0, 0], { stroke: 'teal', width: 2 });
      e.vectors.forEach((q, i) => g2.vector(N.matVec(M, i ? [0, 1] : [1, 0]), { color: 'gold', width: 2.4, head: 9, label: 'q' + SUB[i + 1], labelFont: 'math' }));
      g2.hud('whitening (next): divide each PC by √λ', r.x + 12, r.y + 22, { size: 13.5, color: 'teal', bold: true });
      g2.hud(u < 0.95 ? 'the ellipse becomes a circle…' : 'round: every direction has spread 1', r.x + 12, r.y + 42, { size: 12.5, color: 'muted' });
    });
  }

  /* ---------- recap: three stacked mini-cards ---------- */
  function renderRecap(g, s) {
    const dg = s.dig;
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-1.6, 0, 2.4) }, (g2, r) => {
      const S = covN(s.pts2), e = N.eigSym(S), q1 = e.vectors[0];
      g2.points(s.pts2, { r: 2.4, color: 'data', alpha: 0.6 });
      const pts = []; for (let k = 0; k <= 90; k++) { const a = Math.PI * k / 90, d = [Math.cos(a), Math.sin(a)]; pts.push(N.scale(d, N.dot(d, N.matVec(S, d)))); }
      g2.polyline(pts, { color: 'w', width: 1.4, alpha: 0.7 }); g2.polyline(pts.map(p => [-p[0], -p[1]]), { color: 'w', width: 1.4, alpha: 0.7 });
      g2.lineThrough([0, 0], q1, { color: 'gold', width: 2, alpha: 0.6, len: 4 });
      g2.vector(N.scale(q1, Math.sqrt(e.values[0])), { color: 'gold', width: 2.6, head: 9, label: 'q₁', labelFont: 'math' });
      g2.hud('1 · Line of greatest spread: var(d) = dᵀΣd peaks at q₁', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [1.5, 0, 0], dist: 15, yaw: 0.75, pitch: 0.5 }) }, (g2, r) => {
      const fr = frame3(covN(s.p3)), Q = N.fromCols(fr.q), Qt = N.transpose(Q);
      g2.plane({ point: [0, 0, 0], span: [fr.q[0], fr.q[1]], extentU: 6.5, extentV: 3, color: 'gold', alpha: 0.15 });
      const P = s.p3.slice(0, 120);
      P.forEach(p => { const c = N.matVec(Qt, p), f = N.matVec(Q, [c[0], c[1], 0]); g2.point(p, { r: 2, color: 'muted', alpha: 0.4 }); g2.line(p, f, { color: 'residual', dashed: [2, 3], width: 1, alpha: 0.6 }); g2.point(f, { r: 2.4, color: 'data' }); });
      const tot = fr.lam[0] + fr.lam[1] + fr.lam[2];
      g2.bars(fr.lam.map(l => 100 * l / tot), { screen: true, x: r.x + r.w - 150, y: r.y + 60, w: 120, h: 90, max: 100, colors: ['gold', 'gold', 'muted'], labels: ['85 %', '14 %', '1 %'] });
      g2.hud('2 · Rotate, then crop: keep the top k', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-2.4, 0, 2.4) }, (g2, r) => {
      const k = digitScale(s) * 0.9;
      g2.points(dg.Z.map(z => [k * z[0], k * z[1]]), { r: 3, colors: i => CLASS[dg.lab[i]] });
      g2.hud('3 · A shadow carries its % label', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
      g2.hud('shadow of ℝ⁶⁴, ' + Math.round(100 * dg.cum2) + ' % of variance', r.x + 12, r.y + 42, { size: 12.5, color: 'gold' });
    });
  }
})();
