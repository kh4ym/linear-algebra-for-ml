/* p4-12 — Condition number, pseudoinverse via SVD, ridge (κ = thin ellipse; reverse the moves, skip σ = 0; σ/(σ²+λ)). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpM = LA.lerpM, C = LA.color;
  const HONEST = 'The picture stops at 3. The formula does not.';
  const A0 = [[1, 1], [1, 1.1]];
  const B0 = [2, 2.1];
  const BN = [2, 2.12];                                   // the noisy b of step 1
  const SUB = ['₀', '₁', '₂'];
  const goldS = t => '<span class="c-gold">' + t + '</span>';
  const bS = '<span class="c-b">b</span>';

  /* ---------- numerics helpers ---------- */
  function svdA(M) { const r = N.svd(M); return { S: r.S.slice(), u: N.cols(r.U), v: N.cols(r.V) }; }
  /** SVD of the matrix actually used: σ₂ scaled by s.sigma2Scale. */
  function eff(s, M) {
    const d = svdA(M || s.A); const S = [d.S[0], d.S[1] * (M ? 1 : s.sigma2Scale)];
    const Mx = N.addM(N.scaleM(N.outer(d.u[0], d.v[0]), S[0]), N.scaleM(N.outer(d.u[1], d.v[1]), S[1]));
    return { S, u: d.u, v: d.v, M: Mx };
  }
  /** the reverse-move filter: 1/σ (λ = 0), σ/(σ²+λ) (ridge), 0 for σ = 0 (pseudo-inverse). */
  function filt(sig, lam, smax) { if (!(sig > 1e-6 * Math.max(smax, 1e-12))) return 0; return sig / (sig * sig + lam); }
  function pinvL(d, lam) {
    let P = N.zerosM(2, 2);
    for (let i = 0; i < 2; i++) P = N.addM(P, N.scaleM(N.outer(d.v[i], d.u[i]), filt(d.S[i], lam, d.S[0])));
    return P;
  }
  const kappa = d => d.S[1] > 1e-9 * d.S[0] ? d.S[0] / d.S[1] : Infinity;
  const fmtK = k => Number.isFinite(k) ? LA.fmt.fixed(k, 1) : '∞';
  function sqrtSym(S) {
    const e = N.eigSym(S); const Q = N.fromCols(e.vectors);
    return N.matMul(N.matMul(Q, N.diag(e.values.map(l => Math.sqrt(Math.max(l, 0))))), N.transpose(Q));
  }

  // 64 features with planted scales: singular values via eigSym(XᵀX) (computed once, deterministic)
  let SIG64 = null;
  function sig64() {
    if (SIG64) return SIG64;
    const rand = LA.rng(52), n = 200, d = 64, X = [];
    for (let i = 0; i < n; i++) { const r = []; for (let j = 0; j < d; j++) r.push(LA.rng.gauss(rand) * Math.pow(10, -1.5 * j / (d - 1))); X.push(r); }
    const G = N.matMul(N.transpose(X), X);
    const sv = N.eigSym(G).values.map(l => Math.sqrt(Math.max(l, 0)));
    SIG64 = { sv, k: sv[0] / sv[d - 1] };
    return SIG64;
  }

  LA.registerLesson({
    id: 'p4-12-pseudoinverse-conditioning',
    part: 4,
    title: 'Condition number, pseudoinverse via SVD, ridge',
    subtitle: 'A thin ellipse turns small noise into big errors; the pseudo-inverse skips σ = 0, and ridge tames 1/σ into σ/(σ² + λ).',
    minutes: 11,
    prereqs: ['p4-07-svd', 'p3-06-regression-matrix-form', 'p3-01-systems-row-column'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'a 10⁴-column design matrix with raw features can have κ ≈ 10⁶–10⁸; float32 holds about 7 digits', transfers: 'κ = σ_max/σ_min, A⁺ = VΣ⁺Uᵀ and the ridge filter σ/(σ²+λ) are identical for any shape; only the number of σ’s grows' },
    cantShow: ['more than two singular directions as arrows (κ of a 64-column matrix is shown as a bar strip)', 'rounding error itself — we show its amplification, not its origin'],
    terms: [
      { term: 'condition number', def: 'κ = σ_max/σ_min: how much a relative error in the input can grow in the output.' },
      { term: 'ill-conditioned', def: 'Having a large κ: tiny changes in b cause large changes in the solution x.' },
      { term: 'pseudo-inverse', def: 'A⁺ = VΣ⁺Uᵀ: reverse the three SVD moves, inverting each σ > 0 and skipping σ = 0.' },
      { term: 'ridge regression', def: 'Replace 1/σ by σ/(σ²+λ), equivalently solve (XᵀX + λI)w = Xᵀy: shrink the weights, especially along weak directions.' },
      { term: 'standardization', def: 'Rescaling every feature to unit spread so that κ shrinks.' },
    ],
    atlas: ['regularization', 'linear-regression', 'optimization', 'svd'],
    thumbnail: 2,

    state() {
      const pd = LA.rng.planeData(52, 40);
      const Xc = N.center(pd.X).Xc.map(r => r.map(x => Math.round(x * 1e4) / 1e4));
      return {
        view: '2d', cam: cam2(0, 0, 2.6),
        A: LA.clone(A0), b: B0.slice(), nudge: 0, noiseR: 0.1, noiseT: 0, lambda: 0, showInverseEllipse: false,
        stage: 0, sigma2Scale: 1, scaleA: 0, featScale: 1, stdT: 0, dir: [0.72, -0.69], Splus: [[0, 0], [0, 0]],
        X: Xc, scene: 'rows', rowsA: 1, ellT: 0, axesT: 0, showB: false, showX: false, bars: null, cross: false, okMark: false,
        colHandles: false, dirHandle: false, track: null, chart: false, pathT: 0, mlT: 0, panel: 'rows',
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'rows') { renderRows(g, s); return; }
      if (s.scene === 'path') { renderPath(g, s); return; }
      if (s.scene === 'feat') { renderFeat(g, s); return; }
      if (s.scene === 'sigma') { renderSigma(g, s); return; }
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      renderMap(g, s, ctx);
    },

    math(s, tex) {
      const P = []; const tn = (x, c) => tex.text(Number.isFinite(x) ? LA.fmt(x) : '∞', { color: c });
      const A = tex.sym('A'), Sg = tex.sym('Σ'), U = tex.sym('U'), V = tex.sym('V');
      const d = eff(s);
      switch (s.panel) {
        case 'rows': {
          const b = [B0[0], B0[1] + 0.02 * s.nudge], x = N.solve(A0, b);
          P.push(tex.eq(A, tex.sym('x'), tex.op('='), tex.sym('b', { color: 'b' }), tex.op(':'), tex.mat(A0), tex.sym('x'), tex.op('='), tex.vec(b, { color: 'b' })));
          P.push(tex.eq(tex.sym('x'), tex.op('='), tex.vec([1, 1], { color: 'gold' }), tex.op('→'), tex.vec(x, { color: 'gold' })));
          break; }
        case 'kappa':
          P.push(tex.eq(tex.sym('κ'), tex.op('='), tex.frac(tex.sym('σ', { sub: '1' }), tex.sym('σ', { sub: '2' })), tex.op('='), tex.frac(tn(d.S[0], 'gold'), tn(d.S[1], 'gold')), tex.op('≈'), tn(kappa(d), 'gold')));
          break;
        case 'noise':
          P.push(tex.eq(tex.inv(A), tex.op('='), V, tex.inv(Sg), tex.T(U), tex.op(';'), tex.text('relative error ≤ κ · relative noise')));
          P.push(tex.eq(tex.text('long semi-axis'), tex.op('='), tex.text('0.1'), tex.op('/'), tex.sym('σ', { sub: '2' }), tex.op('='), tn(s.noiseR / d.S[1], 'gold')));
          break;
        case 'stages':
          P.push(tex.eq(tex.inv(A), tex.op('='), V, tex.inv(Sg), tex.T(U), tex.op(','), tex.inv(Sg), tex.op('='), tex.text('diag'), tex.paren(tn(1 / d.S[0]) + tex.op(',') + tn(1 / d.S[1], 'bad'))));
          break;
        case 'pinv': {
          const f2 = filt(d.S[1], 0, d.S[0]);
          P.push(tex.eq(tex.sym('Σ⁺'), tex.op('='), tex.text('diag'), tex.paren(tn(1 / d.S[0]) + tex.op(',') + tn(f2, f2 === 0 ? 'ok' : 'bad')), tex.op(','), tex.sym('A⁺'), tex.op('='), V, tex.sym('Σ⁺'), tex.T(U)));
          P.push(tex.eq(tex.sym('A⁺'), tex.op('='), tex.mat(pinvL(d, 0), { colors: 'mag' }), tex.text('σ₂ = ' + LA.fmt(d.S[1]))));
          break; }
        case 'det': {
          const k = Math.pow(1000, s.scaleA);
          P.push(tex.eq(tex.text('det'), tex.paren(tn(Math.round(k)) + A), tex.op('='), tn(N.det(A0) * k * k, 'bad'), tex.op(','), tex.sym('κ'), tex.op('='), tn(kappa(d), 'gold')));
          break; }
        case 'ridge':
          P.push(tex.eq(tex.sym('A', { sub: 'λ' }) + '<sup>+</sup>', tex.op('='), V, tex.text('diag'), tex.paren(tex.frac(tex.sym('σ', { sub: 'i' }), tex.sym('σ', { sub: 'i', sup: '2' }) + tex.op('+') + tex.sym('λ'))), tex.T(U), tex.op('='), tex.inv(tex.paren(tex.T(A) + A + tex.op('+') + tex.sym('λ') + tex.sym('I'))) + tex.T(A)));
          P.push(tex.eq(tex.sym('λ'), tex.op('='), tn(s.lambda, 'gold'), tex.op(':'), tn(filt(d.S[0], s.lambda, d.S[0]), 'gold'), tex.op(','), tn(filt(d.S[1], s.lambda, d.S[0]), 'gold')));
          break;
        case 'path': {
          const lam = pathLam(s.pathT), xb = N.matVec(pinvL(eff(s, A0), lam), BN);
          P.push(tex.eq(tex.sym('x', { sub: 'λ' }), tex.op('='), tex.vec(xb, { color: 'w' }), tex.op(','), tex.sym('λ'), tex.op('='), tn(lam, 'gold'), tex.op(';'), tex.sym('x', { sub: 'λ' }), tex.op('→'), tex.text('0 as λ → ∞')));
          break; }
        case 'feat': {
          const k = featKappa(s);
          P.push(tex.eq(tex.sym('κ') + tex.paren(tex.sym('X') + tex.sym('D')), tex.op('='), tn(k, k > 10 ? 'bad' : 'gold'), tex.op(','), tex.sym('D'), tex.op('='), tex.text('diag(1, 1/' + LA.fmt(Math.round(s.featScale)) + ')')));
          P.push(tex.eq(tex.text('64 features: κ = σ₁/σ₆₄ ='), tn(sig64().k, 'bad')));
          break; }
        case 'sigma':
          P.push(tex.eq(Sg, tex.op('='), tex.mat([[2, 0], [0, 0]]), tex.op(','), tex.sym('Σ⁺'), tex.op('='), tex.mat(s.Splus, { colors: 'mag' })));
          break;
        case 'ml':
          P.push(tex.eq(tex.sym('κ'), tex.op('='), tex.sym('σ', { sub: 'max' }), tex.op('/'), tex.sym('σ', { sub: 'min' })));
          break;
        case 'recap':
          P.push(tex.eq(tex.sym('A⁺'), tex.op('='), V, tex.sym('Σ⁺'), tex.T(U), tex.op(','), tex.frac(tex.sym('σ'), tex.sym('σ') + '<sup>2</sup>' + tex.op('+') + tex.sym('λ'))));
          break;
        default: { // play
          const Pl = pinvL(d, s.lambda), dd = N.normalize(s.dir);
          P.push(tex.eq(A, tex.op('='), tex.mat(d.M, { colors: 'cols' }), tex.sym('κ'), tex.op('='), tn(kappa(d), 'gold'), tex.text('det'), tex.op('='), tn(N.det(d.M))));
          P.push(tex.eq(tex.sym('σ'), tex.op('='), tn(d.S[0], 'gold'), tex.op(','), tn(d.S[1], 'gold'), tex.op(';'), tex.text('filter'), tn(filt(d.S[0], s.lambda, d.S[0])), tex.op(','), tn(filt(d.S[1], s.lambda, d.S[0]))));
          P.push(tex.eq(tex.text('amplification of your direction'), tex.op('='), tn(N.norm(N.matVec(Pl, dd)), 'b')));
        }
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Two nearly parallel lines', duration: 4,
        narration: 'Back to the row picture: x + y = 2 and x + 1.1y = 2.1 cross at [1; 1]. The two lines are almost parallel. Change the second right-hand side by 1 % — 2.1 to 2.12 — and the crossing leaps to [0.8; 1.2]: each coordinate moves 20 % from a 1 % change in the data.',
        setup(s) { reset(s, { scene: 'rows', panel: 'rows', nudge: 0 }); s.cam = cam2(1, 1, 1.45); },
        at(s, t) { s.nudge = seg(t, 0.25, 0.85); } },

      { title: 'The same matrix as a thin ellipse', duration: 4,
        narration: 'Look at A as a map: the unit circle becomes a very thin ellipse, σ₁ ≈ 2.05 long and σ₂ ≈ 0.05 thin. The ratio σ₁/σ₂ ≈ 42 is the <b>condition number</b> κ. A round ellipse has κ = 1; a flat one has κ = ∞. Thin means A nearly destroys one direction — and the inverse must nearly create it.',
        setup(s) { reset(s, { scene: 'map', panel: 'kappa', rowsA: 1 }); },
        at(s, t) { s.rowsA = 1 - 0.8 * seg(t, 0, 0.3); s.ellT = seg(t, 0.2, 0.75); s.axesT = seg(t, 0.7, 1); } },

      { title: 'Noise in b becomes a long error in x', duration: 5,
        narration: 'Suppose the orange ' + bS + ' is only known to within a small disk of radius 0.1. Run the inverse: the disk becomes an ellipse of answers, stretched by 1/σ₂ ≈ 20 along ' + goldS('v₂') + ' and squeezed by 1/σ₁ along v₁. The answer could lie anywhere on a sliver four units long. Large κ makes A <b>ill-conditioned</b>: its inverse amplifies noise.',
        setup(s) { reset(s, { scene: 'map', panel: 'noise', rowsA: 0.2, ellT: 0, axesT: 0, showB: true, showX: true, noiseT: 0 }); s.cam = cam2(1.1, 1.0, 2.5); },
        at(s, t) { s.noiseT = seg(t, 0.3, 1); } },

      { title: 'Inverting is reversing the three moves', duration: 5,
        narration: 'A = UΣVᵀ. To undo it, undo each move in reverse order: Uᵀ (a rotation back), then stretch by 1/σ₁ and 1/σ₂, then V. The dangerous step is the middle one: 1/σ₂ = 20.5 is where the sliver comes from. The bars of 1/σ show it at a glance.',
        setup(s) { reset(s, { scene: 'map', panel: 'stages', ellT: 1, axesT: 1, rowsA: 0, stage: 0, bars: 'inv' }); },
        at(s, t) { s.stage = 3 * seg(t, 0.08, 0.92, 'linear'); } },

      { title: 'σ = 0: skip it — the pseudo-inverse', duration: 5, interactive: true,
        narration: 'Push σ₂ all the way to 0. The ellipse goes flat; nothing can be undone along ' + goldS('u₂') + '. The <b>pseudo-inverse</b> simply refuses: Σ⁺ puts 0 where σ was 0, inverts the rest, and A⁺ = VΣ⁺Uᵀ. For a tall X this is (XᵀX)⁻¹Xᵀ, for a wide X it is Xᵀ(XXᵀ)⁻¹ — p3-06’s two formulas are one. Drag the σ₂ track.',
        setup(s) { reset(s, { scene: 'map', panel: 'pinv', ellT: 1, axesT: 1, rowsA: 0, bars: 'inv', showB: true, showX: true, track: 's2' }); s.cam = cam2(0.7, 0.7, 2.7); },
        at(s, t) { s.sigma2Scale = 1 - seg(t, 0.1, 0.75); } },

      { title: 'Misconception: det ≠ 0 means the answer is reliable', duration: 4, misconception: true,
        narration: '✕ “The determinant is 0.1, not zero, so the system is fine.” Multiply A by 1000: det becomes 100 000 — yet the ellipse has exactly the same shape and κ is still 42, so relative errors grow just as much. The determinant measures area; κ measures shape. Only κ predicts reliability.',
        setup(s) { reset(s, { scene: 'map', panel: 'det', ellT: 1, axesT: 1, rowsA: 0, scaleA: 0 }); },
        at(s, t) { s.scaleA = seg(t, 0.1, 0.6); s.cross = t > 0.65; s.okMark = t > 0.8; } },

      { title: 'Ridge: tame 1/σ into σ/(σ² + λ)', duration: 5, interactive: true,
        narration: '<b>Ridge regression</b> changes the dangerous middle move: instead of 1/σ use σ/(σ² + λ). For big σ that is almost 1/σ; for tiny σ it is almost σ/λ — small, not huge. At λ = 0.1 the 1/σ₂ bar drops from 20.5 to 0.48, and the four-unit sliver of answers shrinks to a small blob of radius about 0.05. Drag λ.',
        setup(s) { reset(s, { scene: 'map', panel: 'ridge', ellT: 0.0, axesT: 0, rowsA: 0, bars: 'inv', showB: true, showX: true, noiseT: 1, chart: true, track: 'lam' }); s.cam = cam2(1.1, 1.0, 2.5); },
        at(s, t) { s.lambda = 0.1 * seg(t, 0.1, 0.8); } },

      { title: 'What ridge does to the answer', duration: 4,
        narration: 'The price: the ridge answer is pulled toward the origin, mostly along ' + goldS('v₂') + ', the direction the data barely determines. For the noisy b from the first step, x<sub>λ</sub> slides from the exact [0.8; 1.2] back toward the line of v₁, then slowly toward 0. A little bias for a lot less variance — p5-04’s regularization, seen from the SVD side.',
        setup(s) { reset(s, { scene: 'path', panel: 'path', pathT: 0 }); s.cam = cam2(0.55, 0.6, 1.25); },
        at(s, t) { s.pathT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Where κ comes from: feature scales', duration: 5,
        narration: 'In a design matrix, κ is often just units. Measure column 2 in meters, not centimeters: its numbers shrink 100×, X’s ellipse flattens and κ jumps from 1.2 to 123. <b>Standardization</b> — dividing each column by its spread — rounds the ellipse out and κ falls back near 1. For 64 features κ is σ₁/σ₆₄ on a bar strip. The picture stops at 3. The formula does not.',
        setup(s) { reset(s, { scene: 'feat', panel: 'feat', featScale: 1, stdT: 0 }); s.cam = cam2(0, -1.2, 3.6); },
        at(s, t) { s.featScale = Math.pow(100, seg(t, 0.05, 0.55)); s.stdT = seg(t, 0.65, 0.9); } },

      { title: 'Why ML cares', duration: 4, mlTie: true,
        narration: 'Three habits explained by one ratio: standardize features (κ of X), add ridge or weight decay (lift small σ), and watch the learning rate — gradient descent zigzags on a bowl whose Hessian has large κ, the next part’s capstone. And float32 keeps about 7 digits, so κ ≈ 10⁷ leaves none.',
        setup(s) { reset(s, { scene: 'ml', panel: 'ml', mlT: 0 }); s.cam = cam2(0, 0, 3.6); },
        at(s, t) { s.mlT = seg(t, 0.1, 0.85); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'κ = σ₁/σ₂ is how thin the ellipse is; the inverse stretches noise by 1/σ₂ along ' + goldS('v₂') + ', so relative error grows up to κ×. Invert by reversing the three moves; skip σ = 0 for the pseudo-inverse A⁺ = VΣ⁺Uᵀ. Ridge replaces 1/σ with σ/(σ² + λ). Standardize to keep κ small. det says nothing about it.',
        cards: ['<b>κ = thin ellipse.</b> σ₁/σ₂ ≈ 42: relative error in x can be up to 42× that in b.', '<b>Reverse the moves, skip σ = 0.</b> A⁺ = VΣ⁺Uᵀ inverts each σ > 0 and puts 0 where σ = 0.', '<b>Ridge: σ/(σ² + λ).</b> The tall 1/σ₂ bar collapses and the noise sliver shrinks to a blob.'],
        setup(s) { reset(s, { scene: 'recap', panel: 'recap' }); s.cam = cam2(0, 0, 3.6); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the green and red landing spots (the columns of A), or edit A. Watch the ellipse, κ, det, the σ, 1/σ and ridge bars, and the inverse-noise ellipse around the answer. Drag the orange handle at b to choose a perturbation direction and read its amplification.</p>',
      setup(s) { reset(s, { scene: 'map', panel: 'play', ellT: 1, axesT: 1, rowsA: 0, bars: 'inv', showB: true, showX: true, noiseT: 1, colHandles: true, dirHandle: true, showInverseEllipse: true }); s.cam = cam2(0.9, 0.8, 2.7); },
      controls: [
        { type: 'matrix', bind: 'A', label: 'A', colors: 'cols', step: 0.05 },
        { type: 'slider', bind: 'noiseR', min: 0, max: 0.5, step: 0.01, label: 'noise radius', color: 'b' },
        { type: 'slider', bind: 'lambda', min: 0, max: 1, step: 0.005, label: 'ridge λ', color: 'gold' },
        { type: 'toggle', bind: 'showInverseEllipse', label: 'inverse-noise ellipse' },
        { type: 'slider', bind: 'sigma2Scale', min: 0, max: 1, step: 0.01, label: 'σ₂ scale', color: 'gold' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'κ = 42 [[1,1],[1,1.1]]', values: { A: A0, sigma2Scale: 1, lambda: 0 } },
          { label: 'Round (rotation)', values: { A: [[0.8, -0.6], [0.6, 0.8]], sigma2Scale: 1, lambda: 0 } },
          { label: 'Flat (σ₂ = 0)', values: { A: A0, sigma2Scale: 0, lambda: 0 } },
          { label: 'Feature scale ×100', values: { A: [[1, 0], [0, 0.01]], sigma2Scale: 1, lambda: 0 } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the landing spots (the green and red column tips) until the condition number exceeds 20.',
        setup(s) { reset(s, { scene: 'map', panel: 'kappa', ellT: 1, axesT: 1, rowsA: 0, colHandles: true }); s.A = [[1, 0], [0, 1]]; s.cam = cam2(0.5, 0.5, 2.6); },
        judge(s) { const d = svdA(s.A), k = kappa(d); const ok = k > 20 && d.S[0] > 0.5; return { ok, progress: Math.min(1, Math.log(Math.max(1, k)) / Math.log(20)) * (d.S[0] > 0.5 ? 1 : 0.5), hint: d.S[0] <= 0.5 ? 'Keep the ellipse at least half a unit long; just make it thin.' : undefined }; },
        hint: 'Make the two columns nearly parallel.',
        solution(s) { s.A = LA.clone(A0); } },
      { kind: 'manipulate',
        prompt: 'Drag the λ handle to add just enough ridge that the inverse-noise ellipse’s long semi-axis stays under 0.3 (noise radius 0.1).',
        setup(s) { reset(s, { scene: 'map', panel: 'ridge', rowsA: 0, bars: 'inv', showB: true, showX: true, noiseT: 1, track: 'lam' }); s.lambda = 0; s.noiseR = 0.1; s.cam = cam2(1.1, 1.0, 2.5); },
        judge(s) { const d = svdA(s.A); const m = Math.max(...d.S.map(x => s.noiseR * filt(x, s.lambda, d.S[0]))); return { ok: m < 0.3, progress: LA.clamp(1 - (m - 0.3) / 1.8, 0, 1) }; },
        hint: 'Watch the tall bar — λ needs to beat σ₂² ≈ 0.0024 by a wide margin.',
        solution(s) { s.lambda = 0.05; } },
      { kind: 'entry',
        prompt: 'The gold segment is the image of the unit circle under Σ = diag(2, 0). Type its pseudo-inverse Σ⁺; the magenta answer x = Σ⁺b follows what you type.',
        bind: 'Splus', answer: [[0.5, 0], [0, 0]], tol: LA.TOL.entry, label: 'Σ⁺', colors: 'mag',
        setup(s) { reset(s, { scene: 'sigma', panel: 'sigma' }); s.Splus = [[0, 0], [0, 0]]; s.cam = cam2(0.3, 0, 2.2); } },
    ],
  });

  /* ---------- helpers ---------- */
  function cam2(x, y, h) { return Object.assign(LA.cam.defaults2d(), { center: [x, y], halfH: h }); }
  function reset(s, o) {
    Object.assign(s, {
      view: '2d', A: LA.clone(A0), b: B0.slice(), nudge: 0, noiseR: 0.1, noiseT: 0, lambda: 0, showInverseEllipse: false,
      stage: 0, sigma2Scale: 1, scaleA: 0, featScale: 1, stdT: 0, rowsA: 0, ellT: 0, axesT: 0, showB: false, showX: false,
      bars: null, cross: false, okMark: false, colHandles: false, dirHandle: false, track: null, chart: false, pathT: 0, mlT: 0,
    }, o || {});
    s.cam = cam2(0, 0, 2.6);
  }
  function pathLam(u) { return u <= 0 ? 0 : Math.pow(10, -4 + 4 * u); }
  function rowLine(g, n, c, opts) { const nn = N.dot(n, n); const p = N.scale(n, c / nn); g.lineThrough(p, [-n[1], n[0]], Object.assign({ len: 12 }, opts)); }
  function featPts(s) {
    const sd = [0, 1].map(j => Math.sqrt(s.X.reduce((a, r) => a + r[j] * r[j], 0) / s.X.length));
    return s.X.map(r => LA.lerpV([r[0], r[1] / s.featScale], [r[0] / sd[0], r[1] / sd[1]], s.stdT));
  }
  function featKappa(s) { const d = svdA(N.matMul(N.transpose(featPts(s)), featPts(s))); return Math.sqrt(d.S[0] / Math.max(d.S[1], 1e-300)); }

  /** σ, 1/σ (or ridge) bars as a HUD at (x, y). */
  function drawBars(g, d, lam, x, y, opts) {
    opts = opts || {};
    const f = [filt(d.S[0], lam, d.S[0]), filt(d.S[1], lam, d.S[0])];
    const cap = 22, h = opts.h || 120;
    const vals = [d.S[0], d.S[1], Math.min(f[0], cap), Math.min(f[1], cap)];
    const ridge = lam > 1e-6;
    g.bars(vals, { screen: true, x, y, w: opts.w || 170, h, max: cap, gap: 8, colors: ['gold', 'gold', ridge ? 'gold' : 'muted', ridge ? 'gold' : 'muted'], labels: ['σ₁', 'σ₂', ridge ? 'f(σ₁)' : '1/σ₁', ridge ? 'f(σ₂)' : '1/σ₂'], title: opts.title || (ridge ? 'σ and the ridge filter σ/(σ²+λ)' : 'σ and 1/σ') });
    const bw = ((opts.w || 170) - 24) / 4;
    vals.forEach((v, i) => {
      const actual = i < 2 ? d.S[i] : f[i - 2];
      const skipped = i === 3 && actual === 0;
      g.hud(skipped ? 'skipped' : LA.fmt.fixed(actual, 2), x + i * (bw + 8), y + h - v / cap * h - 10, { size: 11.5, color: skipped ? 'ok' : (i === 3 && !ridge ? 'bad' : 'hud') });
    });
  }

  /* ---------- step 1: rows ---------- */
  function renderRows(g, s) {
    const b2 = B0[1] + 0.02 * s.nudge, x = N.solve(A0, [B0[0], b2]);
    g.ghostGrid(); g.axes();
    rowLine(g, [1, 1], 2, { color: 'teal', width: 2.2 });
    rowLine(g, [1, 1.1], 2.1, { color: 'w', width: 1.2, dashed: true, alpha: s.nudge > 0 ? 0.5 : 0 });
    rowLine(g, [1, 1.1], b2, { color: 'w', width: 2.2 });
    g.label('x + y = 2', [0.2, 1.8], { color: 'teal', size: 13.5, font: 'ui', dx: -20, dy: -14 });
    g.label('x + 1.1y = ' + LA.fmt.fixed(b2, 2), [1.75, 0.3], { color: 'w', size: 13.5, font: 'ui', dy: 18 });
    if (s.nudge > 0) {
      g.line([1, 1], x, { color: 'gold', width: 2, dashed: true, alpha: 0.8 });
      g.point([1, 1], { r: 5, color: 'muted', label: '[1; 1]', labelColor: 'muted' });
      if (s.nudge > 0.98) g.dim([1, 1], x, { label: 'Δx = ' + LA.fmt.fixed(N.dist([1, 1], x), 2), color: 'gold', offset: 18 });
    }
    g.point(x, { r: 7, color: 'gold', label: 'x = [' + LA.fmt.fixed(x[0], 2) + '; ' + LA.fmt.fixed(x[1], 2) + ']', labelColor: 'gold', dx: 12, dy: 18 });
    g.hud('b₂: 2.10 → ' + LA.fmt.fixed(b2, 2) + '   (Δb = ' + LA.fmt.fixed(b2 - 2.1, 2) + ', about 1 %)', 16, 24, { size: 14, color: 'b', bold: true });
  }

  /* ---------- the map scene (steps 2–7, playground, checks 1–2) ---------- */
  function renderMap(g, s, ctx) {
    const k = Math.pow(1000, s.scaleA);
    const d = eff(s), Mdisp = d.M;                       // the picture is drawn for A (1000A rescaled to fit looks identical)
    const sr = N.svd2rot(s.A);
    g.ghostGrid(); g.axes();
    if (s.rowsA > 0.01) { rowLine(g, [1, 1], 2, { color: 'teal', width: 1.8, alpha: s.rowsA }); rowLine(g, [1, 1.1], 2.1, { color: 'w', width: 1.8, alpha: s.rowsA }); }
    // circle → ellipse, and the reverse moves (stage 0..3)
    if (s.ellT > 0) {
      let M = lerpM([[1, 0], [0, 1]], Mdisp, s.ellT);
      if (s.stage > 0) {
        const u1 = LA.clamp(s.stage, 0, 1), u2 = LA.clamp(s.stage - 1, 0, 1), u3 = LA.clamp(s.stage - 2, 0, 1);
        const D = N.diag([LA.lerp(1, 1 / sr.S[0], u2), LA.lerp(1, 1 / sr.S[1], u2)]);
        M = N.matMul(N.rot2(sr.thetaV * u3), N.matMul(D, N.matMul(N.rot2(-sr.thetaU * u1), Mdisp)));
        const lab = s.stage < 1 ? 'Uᵀ: rotate back' : s.stage < 2 ? 'Σ⁻¹: stretch by 1/σ₁, 1/σ₂' : s.stage < 2.999 ? 'V: rotate' : 'A⁻¹A = I: the unit circle again';
        g.hud(lab, 16, 24, { size: 14, color: s.stage >= 1 && s.stage < 2 ? 'bad' : 'hud', bold: true });
        g.circle([0, 0], 1, { stroke: 'muted', width: 1, dashed: true, strokeAlpha: 0.5 });
      }
      const flat = d.S[1] < 1e-3;
      g.ellipse(M, [0, 0], { stroke: flat ? 'gold' : 'teal', width: flat ? 3 : 2.2 });
      if (s.axesT > 0 && s.stage < 0.01) {
        d.u.forEach((u, i) => { const L = d.S[i] * s.axesT; if (L > 0.012) g.vector(N.scale(u, L), { color: 'gold', width: 2.6, head: Math.min(11, 4 + L * 30), label: 'u' + SUB[i + 1], labelFont: 'math' }); });
        g.hud('σ₁ = ' + LA.fmt.fixed(d.S[0], 2) + '   σ₂ = ' + LA.fmt.fixed(d.S[1], 3) + '   κ = ' + fmtK(kappa(d)), 16, s.stage > 0 ? 46 : 24, { size: 14, color: 'gold', bold: true });
      }
    }
    // b, its noise disk, and the answers
    const Pl = pinvL(d, s.lambda), b = s.b, xl = N.matVec(Pl, b);
    if (s.showB) {
      g.vector(b, { color: 'b', width: 2.6, label: 'b', labelFont: 'math' });
      if (s.noiseT > 0 || s.panel === 'play') {
        g.circle(b, s.noiseR, { stroke: 'b', width: 1.2, dashed: true });
        const u = s.panel === 'play' ? 1 : s.noiseT;
        for (let i = 0; i < 24; i++) {
          const a = 2 * Math.PI * i / 24, dl = [s.noiseR * Math.cos(a), s.noiseR * Math.sin(a)];
          const p0 = N.add(b, dl), p1 = N.matVec(Pl, p0);
          g.point(LA.lerpV(p0, p1, u), { r: 2.6, color: 'muted', alpha: 0.9 });
        }
        if ((s.noiseT > 0.95 || s.showInverseEllipse) && s.noiseR > 0) {
          g.ellipse(N.scaleM(Pl, s.noiseR), xl, { stroke: 'gold', width: 1.8, fill: 'gold', alpha: 0.12 });
          const v2 = d.v[1], semi = s.noiseR * filt(d.S[1], s.lambda, d.S[0]);
          if (s.panel === 'noise') { g.vector(xl, N.add(xl, N.scale(v2, semi)), { color: 'gold', width: 2.4, head: 10, label: 'v₂', labelFont: 'math' }); g.label('×1/σ₂ ≈ ' + LA.fmt.fixed(1 / d.S[1], 1) + ': semi-axis ' + LA.fmt.fixed(semi, 2), N.add(xl, N.scale(v2, semi)), { color: 'gold', size: 12.5, font: 'ui', dx: -16, dy: 18, anchor: 'right' }); }
          if (s.panel === 'ridge' || s.panel === 'play') g.hud('long semi-axis of the answer ellipse: ' + LA.fmt.fixed(Math.max(s.noiseR * filt(d.S[0], s.lambda, d.S[0]), semi), 2), 16, s.axesT > 0 ? 46 : 24, { size: 14, color: 'gold', bold: true });
        }
      }
    }
    if (s.showX) {
      const isP = d.S[1] < 1e-6 || s.panel === 'pinv';
      if (s.panel === 'pinv' && d.S[1] < 1e-6) {
        const bp = N.scale(d.u[0], N.dot(b, d.u[0]));
        g.line(b, bp, { color: 'residual', dashed: true, width: 1.4 });
        g.point(bp, { r: 3.5, color: 'b' });
      }
      // small in the ridge step so the tiny answer blob stays visible
      g.point(xl, { r: s.panel === 'ridge' ? 3 : 6.5, color: isP ? 'mag' : 'gold', label: isP && d.S[1] < 1e-6 ? 'x⁺ = A⁺b (min-norm)' : (s.lambda > 0 ? 'x_λ' : 'x'), labelColor: isP ? 'mag' : 'gold', dx: -12, dy: -14 });
    }
    if (s.bars === 'inv') drawBars(g, d, s.lambda, 18, 84);
    if (s.chart) {
      const series = [{ points: [], color: 'gold' }, { points: [], color: 'muted', dash: [4, 4] }];
      for (let i = 1; i <= 60; i++) { const x = 2.2 * i / 60; series[0].points.push([x, Math.min(filt(x, s.lambda, 1), 12)]); series[1].points.push([x, Math.min(1 / x, 12)]); }
      d.S.forEach(sg => series.push({ points: [[sg, 0], [sg, Math.min(filt(sg, s.lambda, 1), 12)]], color: 'b' }));
      g.lineChart(series, { x: g.size.w - 200, y: g.size.h - 210, w: 175, h: 105, xRange: [0, 2.2], yRange: [0, 12], title: 'gold σ/(σ²+λ) vs dashed 1/σ', xLabel: 'σ (orange marks: σ₂, σ₁)' });
    }
    // misconception readouts
    if (s.panel === 'det') {
      g.hud('A × ' + LA.fmt(Math.round(k)) + '  (drawn rescaled to fit: same shape)', 16, g.size.h - 64, { size: 13.5, color: 'hud' });
      g.hud('det = ' + LA.fmt(Math.round(N.det(A0) * k * k * 100) / 100), 16, g.size.h - 42, { size: 15, color: 'bad', bold: true });
      g.hud('κ = ' + fmtK(kappa(d)) + '  (unchanged)', 16, g.size.h - 20, { size: 15, color: 'gold', bold: true });
      if (s.cross) { g.label('“det ≠ 0 ⇒ safe”', [g.size.w - 50, 24], { screen: true, anchor: 'right', color: 'bad', size: 14, font: 'ui' }); g.marker.cross([g.size.w - 28, 24], { screen: true, size: 22 }); }
      if (s.okMark) g.marker.check([200, g.size.h - 20], { screen: true, size: 18 });
    }
    // handles and tracks
    if (s.colHandles) {
      const c1 = N.col(s.A, 0), c2 = N.col(s.A, 1);
      g.vector(c1, { color: 'i', width: 2.4, head: 10, label: 'a₁', labelFont: 'math' });
      g.vector(c2, { color: 'j', width: 2.4, head: 10, label: 'a₂', labelFont: 'math' });
      g.handle('c1', c1, (p, st) => { st.A[0][0] = p[0]; st.A[1][0] = p[1]; }, { color: 'i' });
      g.handle('c2', c2, (p, st) => { st.A[0][1] = p[0]; st.A[1][1] = p[1]; }, { color: 'j' });
    }
    if (s.dirHandle && s.showB) {
      const dd = N.normalize(s.dir), hp = N.add(b, N.scale(dd, Math.max(s.noiseR, 0.15)));
      g.line(b, hp, { color: 'b', width: 2 });
      const amp = N.matVec(Pl, N.scale(dd, Math.max(s.noiseR, 0.15)));
      g.vector(xl, N.add(xl, amp), { color: 'b', width: 2, head: 9, alpha: 0.8, label: 'δx', labelFont: 'math' });
      g.handle('dir', hp, (p, st) => { const v = N.sub([p[0], p[1]], st.b); if (N.norm(v) > 1e-3) st.dir = N.normalize(v); }, { color: 'b' });
      g.hud('amplification |A⁺δ|/|δ| = ' + LA.fmt.fixed(N.norm(N.matVec(Pl, dd)), 2), 16, g.size.h - 20, { size: 13.5, color: 'b' });
    }
    if (s.track === 's2') drawTrack(g, s, 'sigma2Scale', 0, 1, 'σ₂ scale');
    if (s.track === 'lam') drawTrack(g, s, 'lambda', 0, 0.2, 'λ');
  }

  function drawTrack(g, s, key, lo, hi, name) {
    const P = g.P, bnd = P.bounds();
    const y = bnd.y0 + 0.35, x0 = bnd.x0 + 0.5, x1 = bnd.x0 + 0.5 + Math.min(3, (bnd.x1 - bnd.x0) * 0.55);
    const X = v => x0 + (x1 - x0) * (v - lo) / (hi - lo);
    g.line([x0, y], [x1, y], { color: 'muted', width: 2, alpha: 0.8 });
    [lo, hi].forEach(v => g.label(LA.fmt(v), [X(v), y], { anchor: 'below', dy: 6, color: 'muted', size: 11, font: 'ui' }));
    g.label(name + ' = ' + LA.fmt.fixed(s[key], key === 'lambda' ? 3 : 2), [x1, y], { anchor: 'left', dx: 12, color: 'gold', size: 13, font: 'ui' });
    const hp = [X(LA.clamp(s[key], lo, hi)), y];
    g.point(hp, { r: 4, color: 'gold' });
    g.handle('trk', hp, (p, st) => { st[key] = LA.clamp(lo + (p[0] - x0) / (x1 - x0) * (hi - lo), lo, hi); }, { color: 'white' });
  }

  /* ---------- step 8: the ridge path in weight space ---------- */
  function renderPath(g, s) {
    const d = eff(s, A0);
    g.ghostGrid(); g.axes();
    d.v.forEach((v, i) => g.lineThrough([0, 0], v, { color: 'gold', width: 1.2, alpha: 0.4, len: 4 }));
    g.label('v₁ line', N.scale(d.v[0], 1.75), { color: 'gold', size: 12, font: 'ui', dx: 26 });
    g.label('v₂ line', N.scale(d.v[1], -0.45), { color: 'gold', size: 12, font: 'ui', dx: -28 });
    const pts = []; for (let k = 0; k <= 80; k++) pts.push(N.matVec(pinvL(d, pathLam(k / 80)), BN));
    const nShow = Math.max(1, Math.round(80 * s.pathT));
    g.polyline(pts, { color: 'w', width: 1.2, alpha: 0.35, dashed: true });
    g.polyline(pts.slice(0, nShow + 1), { color: 'w', width: 2.6 });
    const lam = pathLam(s.pathT), xl = N.matVec(pinvL(d, lam), BN), x0 = N.solve(A0, BN);
    g.circle([0, 0], N.norm(xl), { stroke: 'teal', width: 1, dashed: true, strokeAlpha: 0.5 });
    g.point(x0, { r: 6.5, color: 'gold', label: 'exact x = [0.8; 1.2]', labelColor: 'gold', dx: -150, dy: -12 });
    g.point(N.solve(A0, B0), { r: 4, color: 'muted', label: 'noise-free [1; 1]', labelColor: 'muted', dx: 10, dy: 14 });
    g.vector(xl, { color: 'w', width: 2.6, head: 11, label: 'x_λ', labelFont: 'ui' });
    g.hud('λ = ' + (lam < 0.001 ? lam.toExponential(1) : LA.fmt.fixed(lam, 3)) + '   x_λ = [' + LA.fmt.fixed(xl[0], 2) + '; ' + LA.fmt.fixed(xl[1], 2) + ']', 16, 24, { size: 14, color: 'w', bold: true });
    g.hud('ridge path for λ from 10⁻⁴ to 1 · dashed circle: the L2 size |x_λ|', 16, 44, { size: 12.5, color: 'muted' });
  }

  /* ---------- step 9: feature scales ---------- */
  function renderFeat(g, s) {
    const P = featPts(s), S = N.cov(P, { ddof: 0 }), e = N.eigSym(S), H = sqrtSym(S), k = featKappa(s);
    g.ghostGrid(); g.axes();
    g.points(P, { r: 3.4, color: 'data' });
    g.ellipse(H, [0, 0], { stroke: k > 10 ? 'bad' : 'teal', width: 2.2 });
    e.vectors.forEach((q, i) => { const L = Math.sqrt(Math.max(e.values[i], 0)); if (L > 0.02) g.vector(N.scale(q, L), { color: 'gold', width: 2.6, head: 10, label: 'v' + SUB[i + 1], labelFont: 'math' }); });
    g.hud(s.stdT > 0.5 ? 'standardized: each column ÷ its spread' : 'column 2 in units 1/' + LA.fmt(Math.round(s.featScale)) + ' as large', 16, 24, { size: 14, color: s.stdT > 0.5 ? 'ok' : 'hud', bold: true });
    g.hud('κ(X) = ' + LA.fmt.fixed(k, 1) + '   (40 synthetic rows, LA.rng.planeData seed 52)', 16, 44, { size: 13.5, color: k > 10 ? 'bad' : 'gold' });
    const sv = sig64();
    const x = 16, y = g.size.h - 160, w = g.size.w - 32;
    g.bars(sv.sv, { screen: true, x, y, w, h: 90, gap: 1, color: 'gold', title: '64 synthetic features with planted scales: σ₁ … σ₆₄,  κ = σ₁/σ₆₄ = ' + LA.fmt.fixed(sv.k, 0) });
    g.hud(HONEST, 16, g.size.h - 22, { size: 14, color: 'hud', bold: true });
  }

  /* ---------- check 3: Σ = diag(2, 0) and the typed Σ⁺ ---------- */
  function renderSigma(g, s) {
    g.ghostGrid(); g.axes();
    g.circle([0, 0], 1, { stroke: 'muted', width: 1, dashed: true, strokeAlpha: 0.5 });
    g.line([-2, 0], [2, 0], { color: 'gold', width: 4 });
    g.label('Σ·(unit circle): a flat segment, σ = 2 and 0', [0, 0], { color: 'gold', size: 13, font: 'ui', dy: -22 });
    const bb = [1.2, 0.6], x = N.matVec(s.Splus, bb);
    g.vector(bb, { color: 'b', width: 2.4, label: 'b', labelFont: 'math' });
    g.line(bb, [bb[0], 0], { color: 'residual', dashed: true, width: 1.2 });
    g.point(x, { r: 7, color: 'mag', label: 'x = Σ⁺b = [' + LA.fmt.fixed(x[0], 2) + '; ' + LA.fmt.fixed(x[1], 2) + ']', labelColor: 'mag', dx: 12, dy: 22 });
    g.hud('Σ can only reach the x-axis: the part of b off it is skipped.', 16, 24, { size: 13.5, color: 'hud' });
  }

  /* ---------- ML tie-in rows ---------- */
  function renderML(g, s) {
    const u = s.mlT;
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-1.8, 0, 2.2) }, (g2, r) => {
      const M = N.matMul(N.rot2(0.5), N.diag([1.8, LA.lerp(0.06, 1.2, u)]));
      g2.ellipse(M, [0, 0], { stroke: u > 0.9 ? 'teal' : 'bad', width: 2.2 });
      g2.hud('standardize features', r.x + 12, r.y + 22, { size: 13.5, color: 'teal', bold: true });
      g2.hud('κ(X): ' + LA.fmt.fixed(1.8 / LA.lerp(0.06, 1.2, u), 1), r.x + 12, r.y + 42, { size: 12.5, color: u > 0.9 ? 'gold' : 'bad' });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(0, 0, 2.2) }, (g2, r) => {
      const d = svdA(A0), lam = 0.1 * u;
      drawBars(g2, d, lam, r.x + r.w * 0.45, r.y + 50, { w: 200, h: 120 });
      g2.hud('ridge / weight decay', r.x + 12, r.y + 22, { size: 13.5, color: 'gold', bold: true });
      g2.hud('lifts the small σ: 1/σ₂ = 20.5 → ' + LA.fmt.fixed(filt(d.S[1], lam, d.S[0]), 2), r.x + 12, r.y + 42, { size: 12.5, color: 'muted' });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-1.2, 0, 1.5) }, (g2, r) => {
      const a = 1, bq = 25;
      [0.3, 1, 2, 3.5].forEach(c => g2.ellipse([[Math.sqrt(2 * c / a), 0], [0, Math.sqrt(2 * c / bq)]], [0, 0], { stroke: 'teal', width: 1.2, strokeAlpha: 0.7 }));
      let p = [-2.6, 0.8]; const pts = [p.slice()]; const n = Math.round(30 * u);
      for (let i = 0; i < n; i++) { p = [p[0] - 0.075 * a * p[0], p[1] - 0.075 * bq * p[1]]; pts.push(p.slice()); }
      g2.polyline(pts, { color: 'v', width: 2 }); g2.point(pts[pts.length - 1], { r: 4, color: 'v' });
      g2.label('κ = 25', [1.4, 0.95], { color: 'bad', size: 13, font: 'ui', bold: true });
      g2.hud('gradient descent zig-zags (p5-03)', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
      g2.hud('large κ of the Hessian = a long, narrow valley', r.x + 12, r.y + 42, { size: 12.5, color: 'muted' });
    });
  }

  /* ---------- recap: three stacked mini-cards ---------- */
  function renderRecap(g, s) {
    const d = svdA(A0);
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-1, 0, 1.6) }, (g2, r) => {
      const M = N.addM(N.scaleM(N.outer(d.u[0], d.v[0]), d.S[0]), N.scaleM(N.outer(d.u[1], d.v[1]), d.S[1]));
      g2.ellipse(M, [0, 0], { stroke: 'teal', width: 2.2 });
      g2.vector(N.scale(d.u[0], d.S[0]), { color: 'gold', width: 2.4, head: 9, label: 'u₁', labelFont: 'math' });
      g2.hud('1 · κ = σ₁/σ₂ ≈ ' + LA.fmt.fixed(kappa(d), 0) + ': a thin ellipse', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(0, 0, 2) }, (g2, r) => {
      const flat = { S: [d.S[0], 0], u: d.u, v: d.v };
      drawBars(g2, flat, 0, r.x + r.w * 0.45, r.y + 50, { w: 200, h: 120, title: 'Σ⁺: invert σ > 0, skip σ = 0' });
      g2.hud('2 · Reverse the moves, skip σ = 0: A⁺ = VΣ⁺Uᵀ', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: cam2(-2.6, 0, 1.4) }, (g2, r) => {
      drawBars(g2, d, 0.1, r.x + r.w * 0.55, r.y + 50, { w: 200, h: 120 });
      const P0 = pinvL(d, 0), P1 = pinvL(d, 0.1), c = [-3.6, -0.1];
      g2.ellipse(N.scaleM(P0, 0.5), c, { stroke: 'muted', width: 1.2, dashed: true });
      g2.ellipse(N.scaleM(P1, 0.5), c, { stroke: 'gold', width: 2, fill: 'gold', alpha: 0.25 });
      g2.label('λ = 0: the sliver (clipped)', [-2.0, -1.0], { color: 'muted', size: 12, font: 'ui', anchor: 'left' });
      g2.label('λ = 0.1: a blob', N.add(c, [0.35, 0.2]), { color: 'gold', size: 12, font: 'ui', anchor: 'left' });
      g2.hud('3 · Ridge: σ/(σ² + λ) — the sliver shrinks to a blob', r.x + 12, r.y + 22, { size: 13.5, color: 'hud', bold: true });
    });
  }
})();
