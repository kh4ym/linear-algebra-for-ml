/* p4-02 — Eigenvectors and eigenvalues: the directions a matrix only stretches. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, C = LA.color;
  const I2 = () => [[1, 0], [0, 1]];
  const A0 = [[3, 1], [0, 2]];                   // non-symmetric workhorse: λ = 3 on î, λ = 2 on [1; −1]
  const SYM = [[2, 1], [1, 2]];                  // symmetric workhorse: λ = 3 on [1; 1], λ = 1 on [1; −1]
  const ROT = [[0, -1], [1, 0]];
  const SPIN = 20 * Math.PI / 180;
  const SPIRAL = N.scaleM(N.rot2(SPIN), 1.1);
  const SHEAR = [[1, 1], [0, 1]];
  const AX3 = N.normalize([1, 1, 0.5]);
  const FAN = Array.from({ length: 24 }, (_, k) => [Math.cos(k * Math.PI / 12), Math.sin(k * Math.PI / 12)]);
  const ihat = '<span class="c-i">î</span>', vS = '<span class="c-v">v</span>';
  const g1 = '<span class="c-gold">[1; 1]</span>', g2 = '<span class="c-gold">[1; −1]</span>';

  const detAt = (A, l) => (A[0][0] - l) * (A[1][1] - l) - A[0][1] * A[1][0];
  const lamOf = s => s.lamFromEntry ? s.lamE[0] : s.lambda;
  function motionM(s) {
    if (s.kind === 'rot') return N.scaleM(N.rot2(s.rotAng * s.morph), Math.pow(s.rotScale, s.morph));
    return lerpM(I2(), s.A, s.morph);
  }
  function eigPairs(A) {
    const e = N.eig2(A);
    if (!e.real) return null;
    return e.vectors.map((v, i) => ({ val: e.values[i], vec: v }));
  }
  const onLine = (u, d) => Math.abs(u[0] * d[1] - u[1] * d[0]) < 1e-6;
  const fmtV = v => '[' + LA.fmt(v[0]) + '; ' + LA.fmt(v[1]) + ']';
  function charText(A) {
    const [[a, b], [c, d]] = A;
    if (Math.abs(b) < 1e-9 || Math.abs(c) < 1e-9) return '(' + LA.fmt(a) + ' − λ)(' + LA.fmt(d) + ' − λ)';
    const tr = a + d, dt = a * d - b * c;
    return 'λ² ' + (tr >= 0 ? '− ' : '+ ') + LA.fmt(Math.abs(tr)) + 'λ ' + (dt >= 0 ? '+ ' : '− ') + LA.fmt(Math.abs(dt));
  }
  function eigText(A) {
    const e = N.eig2(A);
    if (!e.real) return 'complex, |λ| = ' + LA.fmt(Math.hypot(e.re, e.im));
    if (e.defective) return 'λ = ' + LA.fmt(e.values[0]) + ' twice, one eigenline';
    return 'λ = ' + LA.fmt(e.values[0]) + ', ' + LA.fmt(e.values[1]);
  }
  /** A^k for fractional k in the power preview: blend the normalized integer powers (pure in k). */
  function powTip(A, u, k) {
    const kf = Math.floor(k), fr = k - kf;
    const a = N.normalize(N.matVec(N.matPow(A, kf), u));
    if (fr < 1e-6) return a;
    const b = N.normalize(N.matVec(N.matPow(A, kf + 1), u));
    return N.normalize(lerpV(a, b, fr));
  }
  function makeML() {
    const cloud = LA.rng.gaussianCloud(49, 60, { mean: [0, 0], cov: [[2, 1.2], [1.2, 1]] });
    const e = N.eigSym(N.cov(cloud));
    return { cloud: cloud.map(p => [+(p[0]).toFixed(3), +(p[1]).toFixed(3)]), pcVec: e.vectors, pcVal: e.values };
  }

  LA.registerLesson({
    id: 'p4-02-eigenvectors',
    part: 4,
    title: 'Eigenvectors and eigenvalues',
    subtitle: 'The directions a matrix only stretches, and how to find them where A − λI collapses.',
    minutes: 12,
    prereqs: ['p4-01-change-of-basis', 'p2-06-determinant', 'p2-05-3d-transformations'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'the covariance of 784-pixel images has 784 eigenvectors; a 4096×4096 recurrent weight has 4096 eigenvalues deciding whether signals explode', transfers: 'Av = λv and det(A − λI) = 0 are the definition in any size; “every rotation has an axis” is true only in 3D' },
    cantShow: ['complex eigenvalues as arrows (we show the det curve never touching zero and the spiral they cause)', 'repeated eigenvalues with too few eigenvectors (a shear in n-D) beyond the 2D shear case', 'eigenvectors of a 4×4 or larger matrix as arrows'],
    terms: [
      { term: 'eigenvector', def: 'A nonzero vector that a matrix keeps on its own line, only stretched or flipped.' },
      { term: 'eigenvalue', def: 'The stretch factor λ on an eigenvector’s line (negative means flipped).' },
      { term: 'eigenline', def: 'The whole line of eigenvectors through the origin: every multiple of one eigenvector.' },
      { term: 'characteristic equation', def: 'det(A − λI) = 0: the condition that A − λI squashes some nonzero vector to zero.' },
      { term: 'complex eigenvalues', def: 'What a 2×2 matrix has when every direction turns: no real eigenline exists, but their size still sets growth.' },
    ],
    atlas: ['pca', 'optimization', 'backprop'],
    thumbnail: 1,

    state() {
      const ml = makeML();
      return {
        view: '2d', cam: { center: [1.3, 0.9], halfH: 5.2 },
        scene: 'plane', panel: 'Av',
        A: LA.clone(A0), kind: 'lin', rotAng: Math.PI / 2, rotScale: 1, morph: 0,
        fanAlpha: 1, fanGold: false, eigAlpha: 0, eigLabels: false,
        v: [1, 2], showV: true, vStyle: 'morph',
        avDemo: 0, mult: 0,
        lambda: 0, lamE: [0], lamFromEntry: false, showLam: false, showDetBar: false, lamRange: [0, 4],
        cross: false, spiralK: -1, ellipse: false, powerK: -1,
        theta3: 0, symLock: false, showCols: false, colHandles: 'none', testU: false,
        mlT: 0, cloud: ml.cloud, pcVec: ml.pcVec, pcVal: ml.pcVal,
      };
    },

    render(g, s, ctx) {
      if (s.scene === '3d') { render3d(g, s); return; }
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      renderPlane(g, s, ctx);
    },

    math(s, tex) {
      const P = [], gold = { color: 'gold' };
      const num = (x, o) => '<span class="tx-cell ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';   // see p4-01: avoids the tex.num re-sync bug
      const vsym = (c) => tex.sym('v', { color: c || 'v' });
      switch (s.panel) {
        case 'Av': {
          const F = motionM(s);
          P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols' })));
          P.push(tex.eq(vsym(), tex.op('='), tex.vec(s.v, { color: 'v' }), tex.text(s.morph > 0.5 ? 'becomes' : 'moving to'), tex.sym('A'), vsym(), tex.op('='), tex.vec(N.matVec(F, s.v), { color: 'v' })));
          break; }
        case 'eig2':
          P.push(tex.eq(tex.sym('A'), tex.vec([1, 0], gold), tex.op('='), tex.vec([3, 0], gold), tex.op('='), num(3), tex.op('·'), tex.vec([1, 0], gold)));
          P.push(tex.eq(tex.sym('A'), tex.vec([1, -1], gold), tex.op('='), tex.vec([2, -2], gold), tex.op('='), num(2), tex.op('·'), tex.vec([1, -1], gold)));
          break;
        case 'def':
          P.push(tex.eq(tex.sym('A'), vsym('gold'), tex.op('='), tex.sym('λ'), vsym('gold')));
          P.push(tex.eq(tex.mat(A0, { colors: 'cols' }), tex.vec([1, -1], gold), tex.op('='), tex.vec([2, -2], gold), tex.op('='), num(2), tex.op('·'), tex.vec([1, -1], gold)));
          break;
        case 'line':
          P.push(tex.eq(tex.sym('A'), tex.paren(tex.sym('c') + vsym('gold')), tex.op('='), tex.sym('c'), tex.sym('A'), vsym('gold'), tex.op('='), tex.sym('λ'), tex.paren(tex.sym('c') + vsym('gold'))));
          P.push(tex.text('every multiple of an eigenvector is one'));
          break;
        case 'char': {
          const l = lamOf(s), d = detAt(s.A, l), hit = Math.abs(d) < 0.05;
          const ep = eigPairs(s.A);
          P.push(tex.eq(tex.sym('A'), tex.op('−'), tex.sym('λ'), tex.sym('I'), tex.op('='), tex.mat(N.subM(s.A, N.scaleM(I2(), l)), { colors: 'cols' }), tex.text('at λ ='), num(l, hit ? gold : {})));
          P.push(tex.eq(tex.text('det'), tex.paren(tex.sym('A') + tex.op('−') + tex.sym('λ') + tex.sym('I')), tex.op('='), tex.text(charText(s.A)), tex.op('='), num(d, hit ? gold : {})));
          if (ep && !s.lamFromEntry) P.push(tex.eq(tex.text('collapses at'), tex.sym('λ'), tex.op('='), num(ep[0].val, Math.abs(l - ep[0].val) < 0.05 ? gold : { color: 'muted' }), tex.op(','), num(ep[ep.length - 1].val, Math.abs(l - ep[ep.length - 1].val) < 0.05 ? gold : { color: 'muted' })));
          break; }
        case 'rot': {
          P.push(tex.eq(tex.sym('R'), tex.op('='), tex.mat(ROT, { colors: 'cols' })));
          P.push(tex.eq(tex.text('det'), tex.paren(tex.sym('R') + tex.op('−') + tex.sym('λ') + tex.sym('I')), tex.op('='), tex.sym('λ', { sup: '2' }), tex.op('+'), num(1), tex.op('='), num(detAt(ROT, s.lambda), { color: 'bad' }), tex.op('>'), num(0)));
          break; }
        case 'spiral': {
          const e = N.eig2(SPIRAL);
          P.push(tex.eq(tex.sym('A'), tex.op('='), num(1.1), tex.op('·'), tex.sym('R'), tex.text('(20°)'), tex.op('='), tex.mat(SPIRAL, { digits: 2 })));
          P.push(tex.eq(tex.sym('λ'), tex.op('='), num(e.re), tex.op('±'), num(e.im), tex.sym('i'), tex.op('='), num(1.1), tex.text('(cos 20° ± i sin 20°)')));
          P.push(tex.eq(tex.abs(tex.sym('λ')), tex.op('='), num(Math.hypot(e.re, e.im), gold), tex.text('k = ' + Math.floor(Math.max(0, s.spiralK)))));
          break; }
        case 'sym':
          P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(SYM, { colors: 'cols' }), tex.op('='), '<span>' + tex.T(tex.sym('A')) + '</span>'));
          P.push(tex.eq(tex.text('eigenvectors'), tex.vec([1, 1], gold), tex.text('(λ = 3)'), tex.vec([1, -1], gold), tex.text('(λ = 1)')));
          P.push(tex.eq(tex.rowvec([1, 1], gold), tex.vec([1, -1], gold), tex.op('='), num(0), tex.text('⟂')));
          break;
        case '3d': {
          const R = N.rot3(AX3, s.theta3);
          P.push(tex.eq(tex.sym('R'), tex.op('='), tex.mat(R, { colors: 'cols', digits: 2 }), tex.text('θ = ' + LA.fmt.deg(s.theta3))));
          P.push(tex.eq(tex.sym('R'), tex.sym('a', gold), tex.op('='), tex.sym('a', gold), tex.text('(λ = 1)'), tex.sym('a', gold), tex.op('='), tex.vec(AX3, { color: 'gold', digits: 2 })));
          break; }
        case 'power': {
          const k = Math.max(0, Math.floor(s.powerK));
          P.push(tex.eq(tex.sym('A', { sup: 'k' }), vsym(), tex.op('≈'), tex.sym('c'), tex.op('·'), num(3), tex.sym('', { sup: 'k' }), tex.op('·'), tex.vec([1, 0], gold)));
          P.push(tex.eq(tex.text('k ='), num(k), tex.text('×3 vs ×2 per step: ratio (2/3)ᵏ ='), num(Math.pow(2 / 3, k))));
          break; }
        case 'ml':
          P.push(tex.eq(tex.sym('Σ'), tex.sym('q', gold), tex.op('='), tex.sym('λ'), tex.sym('q', gold), tex.text('principal components')));
          P.push(tex.eq(tex.sym('H'), tex.text(': learning rate < 2 / λmax')));
          P.push(tex.eq(tex.sym('W', { sup: 'k' }), tex.text(': |λ| > 1 explodes, |λ| < 1 vanishes')));
          break;
        case 'check2': {
          const Au = N.matVec(s.A, [1, 1]);
          P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols' })));
          P.push(tex.eq(tex.sym('A'), tex.vec([1, 1], { color: 'b' }), tex.op('='), tex.vec(Au, { color: 'b' }), tex.text('angle to [1; 1]: ' + LA.fmt.deg(N.angle(Au, [1, 1])))));
          break; }
        case 'play': {
          const l = lamOf(s);
          P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols' }), tex.text(eigText(s.A), { color: N.eig2(s.A).real ? 'gold' : 'bad' })));
          P.push(tex.eq(tex.sym('A'), vsym(), tex.op('='), tex.vec(N.matVec(s.A, s.v), { color: 'v' })));
          P.push(tex.eq(tex.text('det'), tex.paren(tex.sym('A') + tex.op('−') + num(l) + tex.sym('I')), tex.op('='), num(detAt(s.A, l), Math.abs(detAt(s.A, l)) < 0.05 ? gold : {})));
          break; }
        default:
          P.push(tex.eq(tex.sym('A'), vsym('gold'), tex.op('='), tex.sym('λ'), vsym('gold')));
          P.push(tex.eq(tex.text('det'), tex.paren(tex.sym('A') + tex.op('−') + tex.sym('λ') + tex.sym('I')), tex.op('='), num(0)));
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Most vectors are knocked off their line', duration: 3.5,
        narration: 'Apply <b>A</b> = [[3, 1], [0, 2]] to a fan of 24 unit arrows. Nearly every arrow turns as well as stretches: it leaves the line its dashed ghost started on. The yellow ' + vS + ' = [1; 2] swings to Av = [5; 4], a different direction entirely — the thin gray line shows where it used to point.',
        setup(s) { Object.assign(s, { scene: 'plane', view: '2d', panel: 'Av', A: LA.clone(A0), kind: 'lin', morph: 0, fanAlpha: 1, fanGold: false, eigAlpha: 0, eigLabels: false, showV: true, vStyle: 'morph', v: [1, 2] }); s.cam = { center: [1.3, 0.9], halfH: 5.2 }; },
        at(s, t) { s.morph = seg(t, 0.1, 0.9); } },

      { title: 'Some stay put: the eigenvectors', duration: 4,
        narration: 'Two directions do not turn. The gold arrow along ' + ihat + ' is simply tripled: A[1; 0] = [3; 0]. The gold arrow along [1; −1] is doubled: A[1; −1] = [2; −2]. These are the <b>eigenvectors</b> of A, and 3 and 2 are their <b>eigenvalues</b>: the stretch factors.',
        setup(s) { s.panel = 'eig2'; s.showV = false; s.fanGold = true; s.morph = 0; s.eigAlpha = 0; s.eigLabels = false; },
        at(s, t) { s.eigAlpha = seg(t, 0, 0.2); s.morph = seg(t, 0.2, 0.85); s.eigLabels = t > 0.85; } },

      { title: 'Written down: Av = λv', duration: 2.5,
        narration: 'On an eigenvector the matrix acts like a plain number: <b>Av = λv</b>. The Greek letter λ (lambda) is the eigenvalue. The big machine A, on these special inputs, is just “multiply by 3” or “multiply by 2”. Here the gold ' + vS.replace('c-v', 'c-gold') + ' = [1; −1] lands on 2' + vS.replace('c-v', 'c-gold') + '.',
        setup(s) { s.panel = 'def'; s.morph = 1; s.fanAlpha = 0.22; s.eigAlpha = 0.5; s.eigLabels = false; s.fanGold = false; s.avDemo = 0; },
        at(s, t) { s.avDemo = seg(t, 0, 0.45); } },

      { title: 'The line matters, not the length', duration: 3,
        narration: 'Any vector on that gold line is an eigenvector: [2; −2], [−0.5; 0.5], all of them get doubled and stay on the line; on the ' + ihat + ' line everything is tripled. So we speak of an <b>eigenline</b> — a direction — and pick any convenient length when we draw one.',
        setup(s) { s.panel = 'line'; s.avDemo = 0; s.mult = 1; s.fanAlpha = 0; s.eigAlpha = 0.6; s.morph = 0; },
        at(s, t) { s.morph = seg(t, 0.15, 0.85); } },

      { title: 'Finding them: slide λ until A − λI collapses', duration: 6, interactive: true,
        narration: 'Av = λv means (A − λI)v = 0: A − λI must squash v to zero, so its determinant vanishes — the <b>characteristic equation</b> det(A − λI) = 0. Watch λ slide: the grid of A − λI flattens onto one line only at λ = 2 and λ = 3. The magenta arrow it crushes to the origin is the eigenvector. Then drag the λ dot.',
        setup(s) { Object.assign(s, { panel: 'char', mult: 0, fanAlpha: 0, eigAlpha: 0.35, morph: 1, showLam: true, showDetBar: true, lambda: 0, lamRange: [0, 4.5], A: LA.clone(A0), kind: 'lin' }); s.cam = { center: [0.6, -0.4], halfH: 4.8 }; },
        at(s, t) { s.lambda = 4 * seg(t, 0.04, 0.78, 'linear') - seg(t, 0.82, 0.95); } },   // sweep 0 → 4, then settle on the collapse at λ = 3

      { title: 'Misconception: every matrix has real eigenvectors', duration: 5, misconception: true,
        narration: '✕ Rotate the plane by 90°. Every arrow turns — the yellow ' + vS + ' leaves its gray line; not one arrow stays on its own line. Slide λ from −3 to 3: det(R − λI) = λ² + 1 is never zero — the red curve never touches the axis and the grid never collapses. Rotations have <i>no</i> real eigenvectors; their eigenvalues are complex numbers, which we do not draw.',
        setup(s) { Object.assign(s, { panel: 'rot', A: LA.clone(ROT), kind: 'rot', rotAng: Math.PI / 2, rotScale: 1, morph: 0, fanAlpha: 1, fanGold: false, eigAlpha: 0, showLam: false, showDetBar: false, lambda: -3, lamRange: [-3, 3], cross: false, showV: true, vStyle: 'morph', v: [1.7, 0.6] }); s.cam = { center: [0, -0.7], halfH: 3.0 }; },
        at(s, t) {
          s.morph = seg(t, 0, 0.35); s.cross = t > 0.3;
          const ph = t > 0.42; s.showLam = ph; s.showDetBar = ph; s.fanAlpha = ph ? 0.3 : 1;
          s.lambda = -3 + 6 * seg(t, 0.45, 1, 'linear');
        } },

      { title: 'Rotation-scalings spiral', duration: 4,
        narration: 'Mix a rotation with a stretch: A = 1.1·R(20°). Still no eigenline, but the <b>complex eigenvalues</b> have size 1.1, and that size shows: apply A again and again and the yellow arrow spirals outward, 10 % longer at every step. The size of λ controls growth even when λ is complex.',
        setup(s) { Object.assign(s, { panel: 'spiral', A: LA.clone(SPIRAL), kind: 'rot', rotAng: SPIN, rotScale: 1.1, morph: 1, fanAlpha: 0, showLam: false, showDetBar: false, cross: false, spiralK: 0, showV: false }); s.cam = { center: [-0.3, 0.3], halfH: 3.7 }; },
        at(s, t) { s.spiralK = 12 * seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Symmetric matrices: perpendicular eigenvectors', duration: 3.5,
        narration: 'For a <i>symmetric</i> matrix the eigenvectors are always perpendicular. A = [[2, 1], [1, 2]] stretches ' + g1 + ' by 3 and ' + g2 + ' by 1 — a right angle apart — and turns the unit circle into an ellipse whose axes are exactly those two gold directions. Part 4 leans on this constantly.',
        setup(s) { Object.assign(s, { panel: 'sym', A: LA.clone(SYM), kind: 'lin', morph: 0, fanAlpha: 0.45, fanGold: false, eigAlpha: 0, eigLabels: false, spiralK: -1, ellipse: true }); s.cam = { center: [0, 0], halfH: 3.6 }; },
        at(s, t) { s.morph = seg(t, 0.05, 0.6); s.eigAlpha = seg(t, 0.55, 0.8); s.eigLabels = t > 0.8; } },

      { title: 'In 3D a rotation has an axis — only in 3D', duration: 5,
        narration: 'Rotate a cube about a tilted line. Every point turns except those on the gold line: it is an eigenvector with λ = 1. This is special to three dimensions. In 2D a rotation fixes nothing but the origin; in 4D a rotation can turn two planes at once and fix no line at all. <b>Only in 3D does every rotation have an axis.</b>',
        setup(s) { Object.assign(s, { scene: '3d', view: '3d', panel: '3d', theta3: 0, ellipse: false }); s.cam = Object.assign(LA.cam.defaults3d(), { target: [0.4, 0.5, 0.4], dist: 8, yaw: -1.0, pitch: 0.32 }); },
        at(s, t) { s.theta3 = (2 * Math.PI / 3) * seg(t, 0.1, 0.9); LA.cam.tween(s, { yaw: -1.0 }, { yaw: -0.6 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Preview: repeating lines everything up', duration: 4,
        narration: 'Back to A = [[3, 1], [0, 2]]. Apply it six times to the whole fan, rescaling each arrow to length 1 so we can see directions. Everything swings toward the gold ' + ihat + ' line, because ×3 beats ×2 at every step. The next lesson turns that into a method — power iteration — and a reason Markov chains settle down.',
        setup(s) { Object.assign(s, { scene: 'plane', view: '2d', panel: 'power', A: LA.clone(A0), kind: 'lin', morph: 0, fanAlpha: 1, fanGold: true, eigAlpha: 0.9, eigLabels: false, ellipse: false, powerK: 0, showV: false }); s.cam = { center: [0, 0], halfH: 2.3 }; },
        at(s, t) { s.powerK = 6 * seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Why ML cares', duration: 4, mlTie: true,
        narration: 'Three places you will meet eigenvectors soon: the covariance matrix’s eigenvectors are the principal components (p4-10); the Hessian’s eigenvalues set the largest safe learning rate (p5-03); a weight matrix’s eigenvalues decide whether repeated application explodes or vanishes (p5-06). Same gold lines, three meanings.',
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.powerK = -1; s.mlT = 0; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Eigenvectors keep their line; eigenvalues are the stretch. Find them where det(A − λI) = 0 — where the grid collapses. Rotations have none in 2D (complex λ) and an axis in 3D only. Symmetric matrices have perpendicular eigenvectors.',
        cards: ['<b>Keeps its line.</b> Av = λv: on an eigenline the matrix is just “multiply by λ”.', '<b>Collapse at det(A − λI) = 0.</b> Slide λ; the grid of A − λI flattens exactly at the eigenvalues.', '<b>Rotation: none in 2D, an axis in 3D.</b> Symmetric matrices: perpendicular gold axes.'],
        setup(s) { s.scene = 'recap'; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Edit A or drag the landing spots of <span class="c-i">î</span> and <span class="c-j">ĵ</span>; drag the yellow ' + vS + ' to compare it with Av. Gold lines are the live eigenlines. Slide λ and draw A − λI to watch the collapse.</p>',
      setup(s) { Object.assign(s, { scene: 'plane', view: '2d', panel: 'play', A: LA.clone(A0), kind: 'lin', morph: 1, fanAlpha: 0.35, fanGold: false, eigAlpha: 1, eigLabels: true, showV: true, vStyle: 'pair', v: [-1, 1.5], showLam: false, showDetBar: false, lambda: 1, lamRange: [-3, 5], showCols: true, colHandles: 'both' }); s.cam = { center: [0.9, 0.4], halfH: 4.6 }; },
      controls: [
        { type: 'matrix', bind: 'A', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'slider', bind: 'lambda', min: -3, max: 5, step: 0.01, label: 'λ (for A − λI)', color: 'gold', fmt: x => LA.fmt(x) },
        { type: 'toggle', bind: 'showLam', label: 'Draw the grid of A − λI' },
        { type: 'toggle', bind: 'showDetBar', label: 'det(A − λI) curve and bar' },
        { type: 'toggle', bind: 'symLock', label: 'Lock symmetric (a₂₁ = a₁₂)' },
        { type: 'preset', label: 'Presets', items: [
          { label: '[[3,1],[0,2]]', apply(s) { toPlane(s); s.A = LA.clone(A0); } },
          { label: 'Symmetric [[2,1],[1,2]]', apply(s) { toPlane(s); s.A = LA.clone(SYM); } },
          { label: 'Rotation 90° (none)', apply(s) { toPlane(s); s.A = LA.clone(ROT); s.symLock = false; } },
          { label: 'Shear (one line)', apply(s) { toPlane(s); s.A = LA.clone(SHEAR); s.symLock = false; } },
          { label: 'Uniform scale (all)', apply(s) { toPlane(s); s.A = [[1.5, 0], [0, 1.5]]; } },
          { label: '3D rotation axis', apply(s) { s.scene = '3d'; s.view = '3d'; s.panel = '3d'; s.theta3 = 2.1; s.cam = Object.assign(LA.cam.defaults3d(), { target: [0.4, 0.5, 0.4], dist: 8, yaw: -0.7, pitch: 0.32 }); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the yellow ' + vS + ' onto an eigenvector of A = [[3, 1], [0, 2]]: Av must lie on the same line as ' + vS + '.',
        setup(s) { checkBase(s); s.A = LA.clone(A0); s.v = [1, 2]; s.showV = true; s.vStyle = 'pair'; s.panel = 'Av'; s.cam = { center: [1.6, 1.1], halfH: 5.4 }; },
        judge(s) {
          const n = N.norm(s.v); if (n < 0.5) return { ok: false, progress: 0, hint: 'Keep ' + vS + ' longer than 0.5.' };
          const Av = N.matVec(s.A, s.v), a = N.angle(s.v, Av) * 180 / Math.PI, d = Math.min(a, 180 - a);
          return { ok: d < LA.TOL.angleDeg, progress: Math.max(0, 1 - d / 45) };
        },
        hint: 'Av should lie on top of ' + vS + ' — try the directions where the fan did not turn: along ' + ihat + ' or along [1; −1].',
        solution(s) { s.v = [1, -1]; } },
      { kind: 'manipulate',
        prompt: 'Drag the landing spot of <span class="c-j">ĵ</span> (column 2) so that the orange u = [1; 1] becomes an eigenvector: Au must lie on u’s line.',
        setup(s) { checkBase(s); s.A = LA.clone(A0); s.showCols = true; s.colHandles = 'j'; s.testU = true; s.panel = 'check2'; },
        judge(s) {
          const Au = N.matVec(s.A, [1, 1]); if (N.norm(Au) < 0.3) return { ok: false, progress: 0, hint: 'Au has shrunk to almost nothing; move the red tip away.' };
          const a = N.angle(Au, [1, 1]) * 180 / Math.PI, d = Math.min(a, 180 - a);
          return { ok: d < LA.TOL.angleDeg, progress: Math.max(0, 1 - d / 40) };
        },
        hint: 'A[1; 1] = column 1 + column 2 = [3; 0] + column 2 must be a multiple of [1; 1], e.g. column 2 = [−1; 2] gives [2; 2].',
        solution(s) { s.A = [[3, -1], [0, 2]]; } },
      { kind: 'entry',
        prompt: 'For A = [[2, 1], [1, 2]], type the <i>larger</i> λ that collapses the grid of A − λI. The grid updates as you type.',
        bind: 'lamE', answer: [3], tol: LA.TOL.entry, label: 'λ', colors: ['gold'],
        setup(s) { checkBase(s); s.A = LA.clone(SYM); s.lamFromEntry = true; s.lamE = [0]; s.showLam = true; s.showDetBar = true; s.lamRange = [-1, 4.5]; s.eigAlpha = 0.25; s.panel = 'char'; s.cam = { center: [0.4, -0.5], halfH: 4.6 }; } },
    ],
  });

  function toPlane(s) { if (s.scene !== 'plane') { s.scene = 'plane'; s.view = '2d'; s.panel = 'play'; s.cam = { center: [0.9, 0.4], halfH: 4.6 }; } s.kind = 'lin'; s.morph = 1; }
  function checkBase(s) {
    Object.assign(s, { scene: 'plane', view: '2d', kind: 'lin', morph: 1, fanAlpha: 0.25, fanGold: false, eigAlpha: 0, eigLabels: false, showV: false, avDemo: 0, mult: 0, showLam: false, showDetBar: false, cross: false, spiralK: -1, ellipse: false, powerK: -1, showCols: false, colHandles: 'none', testU: false, lamFromEntry: false });
    s.cam = { center: [0.8, 0.3], halfH: 4.6 };
  }

  /* ---------------- the main 2D plane ---------------- */
  function renderPlane(g, s, ctx) {
    if (s.symLock && Math.abs(s.A[1][0] - s.A[0][1]) > 1e-9) s.A[1][0] = s.A[0][1];   // playground "lock symmetric"
    const F = motionM(s), lam = lamOf(s);
    const Lm = N.subM(s.A, N.scaleM(I2(), lam));
    const G = s.showLam ? Lm : F;
    const pairs = eigPairs(s.A);
    g.ghostGrid();
    g.grid(G, { alpha: s.spiralK >= 0 ? 0.45 : 0.9 });

    // gold eigenlines of A
    if (s.eigAlpha > 0.01 && pairs) pairs.forEach(p => g.lineThrough([0, 0], p.vec, { color: 'gold', width: 2, alpha: 0.85 * s.eigAlpha }));

    // fan of unit arrows
    if (s.fanAlpha > 0.01) {
      // under a quarter-turn the 15°-spaced fan maps onto itself, so the end picture would look unmoved: use every 60° instead
      (s.kind === 'rot' ? FAN.filter((u, i) => i % 4 === 0) : FAN).forEach(u => {
        const gold = s.fanGold && pairs && pairs.some(p => onLine(u, p.vec));
        let tip;
        if (s.powerK >= 0) tip = powTip(s.A, u, s.powerK);
        else { tip = N.matVec(F, u); g.vector(u, { color: 'residual', dashed: true, width: 1, head: 5, alpha: 0.4 * s.fanAlpha }); }
        g.vector(tip, { color: gold ? 'gold' : 'muted', width: gold ? 2.6 : 1.5, head: gold ? 10 : 7, alpha: s.fanAlpha * (gold ? 1 : 0.85) });
      });
      if (s.eigLabels && s.fanGold && pairs) pairs.forEach(p => {
        const tip = N.matVec(F, p.vec);
        g.label('×' + LA.fmt(p.val), tip, { color: 'gold', size: 14, dx: p.vec[1] < 0 ? 16 : 0, dy: p.vec[1] < 0 ? 14 : -16, font: 'ui', bold: true });
      });
    }

    // eigenline labels in playground / symmetric step (letter + λ)
    if (s.eigLabels && !s.fanGold && pairs && s.eigAlpha > 0.5) {
      const tag = s.ellipse ? ['q₁', 'q₂'] : ['e₁', 'e₂'];
      pairs.forEach((p, i) => {
        const L = s.ellipse ? Math.abs(p.val) : 1.4;
        g.vector(N.scale(p.vec, L), { color: 'gold', width: 3, head: 11, label: tag[i] + '  λ = ' + LA.fmt(p.val), labelFont: 'ui', labelSize: 13 });
      });
    }
    if (s.ellipse) {
      g.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1.2 });
      g.ellipse(F, [0, 0], { stroke: 'teal', width: 2.2, fill: 'teal', alpha: 0.08 });
      if (s.eigAlpha > 0.5 && pairs) g.arc([0, 0], pairs[0].vec, pairs[1].vec, { r: 0.42, color: 'gold', right: true });
    }

    // the yellow test vector
    if (s.showV) {
      if (s.vStyle === 'morph') {
        g.lineThrough([0, 0], s.v, { color: 'hud', width: 1, alpha: 0.35 });
        g.vector(s.v, { color: 'v', dashed: true, width: 1.8, alpha: 0.5, head: 9, label: s.morph > 0.3 ? 'v' : '', labelFont: 'math' });
        const Fv = N.matVec(F, s.v), mname = s.kind === 'rot' && s.rotScale === 1 ? 'Rv' : 'Av';
        g.vector(Fv, { color: 'v', width: 3.2, label: s.morph > 0.3 ? mname : 'v', labelFont: 'math' });
        if (s.kind === 'rot' && s.morph > 0.05) g.arc([0, 0], s.v, Fv, { r: 1.15, color: 'v', label: 'turned ' + LA.fmt.deg(N.angle(s.v, Fv)) });
      } else {
        const Av = N.matVec(s.A, s.v);
        g.lineThrough([0, 0], s.v, { color: 'v', width: 1, alpha: 0.3 });
        g.vector(Av, { color: 'v', width: 2.2, alpha: 0.6, dashed: true, label: 'Av', labelFont: 'math' });
        g.vector(s.v, { color: 'v', width: 3.2, label: 'v', labelFont: 'math' });
        g.handle('v', s.v, (p, st) => { st.v = p; }, { color: 'v' });
      }
    }

    // step 3: Av = λv on the gold eigenvector
    if (s.avDemo > 0.01) {
      const a = s.avDemo, e = [1, -1], Ae = [2, -2];
      g.vector(Ae, { color: 'gold', width: 3, alpha: 0.55 * a, head: 13, label: 'Av = 2v', labelFont: 'ui', labelSize: 13 });
      g.vector(e, { color: 'gold', width: 4, alpha: a, head: 12, label: 'v', labelFont: 'math' });
      // dimension lines on the side opposite the arrow labels (dim's normal flips with the end order)
      if (a > 0.6) { g.dim(e, [0, 0], { label: '|v|', offset: 16, color: 'gold' }); g.dim(Ae, [0, 0], { label: '2|v|', offset: 46, color: 'gold' }); }
    }

    // step 4: many multiples on each eigenline
    if (s.mult > 0.01) {
      const lists = [{ e: [1, -1], l: 2, cs: [1.6, 1, 0.45, -0.5, -1.2], lab: true }, { e: [1, 0], l: 3, cs: [1.2, 0.6, -0.8], lab: false }];
      lists.forEach(L => L.cs.slice().sort((a, b) => Math.abs(b) - Math.abs(a)).forEach(c => {
        const f = c * (1 + (L.l - 1) * s.morph);
        g.point(N.scale(L.e, c), { r: 3, color: 'gold', alpha: 0.45 });
        const f1 = Math.round(f * 10) / 10;
        g.vector(N.scale(L.e, f), { color: 'gold', width: 2.4, head: 10, alpha: 0.9, label: L.lab ? (f1 < 0 ? '−' : '') + Math.abs(f1) + 'v' : '', labelFont: 'ui', labelSize: 11.5 });
      }));
      g.label('×2 on this line', [1.4, -1.4], { color: 'gold', size: 12.5, font: 'ui', dx: 80 });
      g.label('×3 on this line', [-1.9, 0], { color: 'gold', size: 12.5, font: 'ui', dy: -16 });
    }

    // collapse picture: null-space arrow when det(A − λI) ≈ 0
    if (s.showLam) {
      const d = detAt(s.A, lam);
      if (Math.abs(d) < 0.06 && pairs) {
        const p = pairs.reduce((b, q) => Math.abs(q.val - lam) < Math.abs(b.val - lam) ? q : b, pairs[0]);
        g.lineThrough([0, 0], p.vec, { color: 'gold', width: 2.6, alpha: 0.95 });
        // the flattened grid: every grid line of A − λI now lies on one line (its column space)
        const c0 = N.col(Lm, 0), c1 = N.col(Lm, 1), cs = N.norm(c0) > N.norm(c1) ? c0 : c1;
        if (N.norm(cs) > 0.05) {
          const u = N.normalize(cs), up = u[1] < 0 ? N.scale(u, -1) : u;
          g.lineThrough([0, 0], up, { color: 'white', width: 2.4, alpha: 0.75 });
          g.label('the whole grid of A − λI is flattened onto this line', N.scale(up, 1.1), { color: 'white', size: 12.5, font: 'ui', anchor: 'left', dx: 12 });
        }
        // the eigenvector: A − λI sends it to the origin (drawn magenta, with its landing point)
        g.vector(N.scale(p.vec, 1.8), { color: 'mag', width: 3.2, label: 'v', labelFont: 'math' });
        g.marker.ring([0, 0], { r: 9, color: 'mag' });
        g.label('(A − λI)v = 0: this arrow is squashed to the origin', N.scale(p.vec, 1.8), { color: 'mag', size: 12.5, font: 'ui', dy: 34, dx: 6 });
        g.label('so v is an eigenvector, λ = ' + LA.fmt(p.val), N.scale(p.vec, 1.8), { color: 'gold', size: 12.5, font: 'ui', dy: 52, dx: 6 });
      }
    }
    if (s.showDetBar) drawDetChart(g, s, lam);

    if (s.cross) {
      g.marker.cross([0, 0], { size: 30, dx: 0, dy: 0 });
      g.label('no eigenline', [0, 0], { color: 'bad', size: 14, dy: -30, font: 'ui', bold: true });
    }

    // spiral: one vector hit 12 times by A = 1.1·R(20°)
    if (s.spiralK >= 0) {
      const tipAt = k => N.scale([Math.cos(SPIN * k), Math.sin(SPIN * k)], Math.pow(1.1, k));
      const pts = []; for (let k = 0; k <= s.spiralK + 1e-9; k += 0.1) pts.push(tipAt(k));
      pts.push(tipAt(s.spiralK));
      if (pts.length > 1) g.polyline(pts, { color: 'v', width: 1.6, alpha: 0.7 });
      for (let k = 0; k < Math.floor(s.spiralK + 1e-9); k++) g.vector(tipAt(k), { color: 'v', width: 1.4, head: 7, alpha: 0.28 });
      g.vector([1, 0], { color: 'v', width: 2, head: 9, alpha: 0.7, label: 'v', labelFont: 'math', dy: 14 });
      const kk = Math.floor(s.spiralK + 1e-9);
      g.vector(tipAt(s.spiralK), { color: 'v', width: 3.2, label: kk > 0 ? 'A' + supNum(kk) + 'v' : 'v', labelFont: 'math' });
      g.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1 });
      g.hud('|λ| = 1.1: each hit turns 20° and stretches ×1.1', 14, 22, { color: 'gold', size: 13 });
      g.hud('length after ' + kk + ' hits: 1.1' + supNum(kk) + ' = ' + LA.fmt(Math.pow(1.1, kk)), 14, 42, { color: 'v', size: 13 });
    }

    // columns of A as landing spots (playground / check 2)
    if (s.showCols) {
      const ci = N.col(s.A, 0), cj = N.col(s.A, 1);
      g.vector(ci, { color: 'i', label: 'i', hat: true, width: 2.6 });
      g.vector(cj, { color: 'j', label: 'j', hat: true, width: 2.6 });
      if (s.colHandles === 'both') g.handle('ci', ci, (p, st) => { st.A[0][0] = p[0]; st.A[1][0] = p[1]; if (st.symLock) st.A[0][1] = p[1]; }, { color: 'i' });
      if (s.colHandles !== 'none') g.handle('cj', cj, (p, st) => { st.A[0][1] = p[0]; st.A[1][1] = p[1]; if (st.symLock) st.A[1][0] = p[0]; }, { color: 'j' });
    }
    if (s.testU) {
      const Au = N.matVec(s.A, [1, 1]);
      g.lineThrough([0, 0], [1, 1], { color: 'b', width: 1.2, alpha: 0.45 });
      g.vector(Au, { color: 'b', width: 2.2, dashed: true, alpha: 0.75, label: 'Au', labelFont: 'math' });
      g.vector([1, 1], { color: 'b', width: 3, label: 'u', labelFont: 'math' });
    }
    if (ctx.mode === 'play' && !pairs) g.hud('no real eigenvectors: every direction turns (complex λ)', 14, 22, { color: 'bad', size: 13 });
    if (ctx.mode === 'play' && pairs && pairs.length === 1) g.hud('only one eigenline (repeated λ, too few eigenvectors)', 14, 22, { color: 'gold', size: 13 });
  }
  const SUP = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  function supNum(k) { return String(k).split('').map(c => SUP[c] || c).join(''); }

  /** det(A − λI) as λ slides: curve + current λ marker + a gold bar when |det| ≈ 0 + a draggable λ dot (2D only). */
  function drawDetChart(g, s, lam) {
    const W = g.size.w, H = g.size.h;
    const x = 22, w = Math.min(250, W * 0.45), h = 96, y = H - h - 34;
    const [lo, hi] = s.lamRange;
    const pts = []; let ymin = Infinity, ymax = -Infinity;
    for (let i = 0; i <= 60; i++) { const l = lo + (hi - lo) * i / 60, d = detAt(s.A, l); pts.push([l, d]); ymin = Math.min(ymin, d); ymax = Math.max(ymax, d); }
    ymin = Math.min(ymin, -0.6); ymax = Math.max(ymax, 0.6);
    const real = N.eig2(s.A).real;
    const d = detAt(s.A, lam), hit = Math.abs(d) < 0.05;
    const yr = [ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)];
    const lc = LA.clamp(lam, lo, hi);
    g.lineChart([
      { points: pts, color: real ? 'hud' : 'bad', width: 2 },
      { points: [[lc, yr[0]], [lc, yr[1]]], color: hit ? 'gold' : 'v', width: 1.6, dash: [4, 3] },
    ], { x, y, w, h, xRange: [lo, hi], yRange: yr, hline: 0, hlineColor: 'muted', title: 'det(A − λI) as λ slides', xLabel: 'λ from ' + LA.fmt(lo) + ' to ' + LA.fmt(hi) });
    g.hud('λ = ' + LA.fmt.fixed(lam, 2) + '    det = ' + LA.fmt.fixed(d, 2), x, y - 28, { color: hit ? 'gold' : 'hud', size: 13, bold: hit });
    // |det| bar
    const bmax = Math.max(Math.abs(ymin), Math.abs(ymax));
    g.bars([Math.max(Math.abs(d), bmax * 0.004)], { screen: true, x: x + w + 22, y, w: 18, h, max: bmax, color: hit ? 'gold' : (real ? 'hud' : 'bad') });
    g.hud('|det|', x + w + 16, y - 10, { color: 'muted', size: 11 });
    // zero crossings marked gold on the chart
    const X = l => x + (l - lo) / (hi - lo) * w, Y = v => y + h - (v - yr[0]) / (yr[1] - yr[0]) * h;
    const pairs = eigPairs(s.A);
    if (pairs && !s.lamFromEntry) pairs.forEach(p => { if (p.val >= lo && p.val <= hi) g.label('●', [X(p.val), Y(0)], { screen: true, color: 'gold', size: 11, bg: false }); });
    // draggable λ dot on the zero line
    if (!g.is3d && !s.lamFromEntry) {
      const P = g.P, wpos = P.toWorld(X(lc), Y(0));
      g.handle('lam', wpos, (p, st) => { const sx = P.toScreen(p)[0]; st.lambda = LA.clamp(lo + (sx - x) / w * (hi - lo), lo, hi); }, { color: 'v', r: 12 });
    }
  }

  /* ---------------- 3D: a rotation and its axis ---------------- */
  function render3d(g, s) {
    const R = N.rot3(AX3, s.theta3);
    g.cage({ range: 2, walls: false });
    g.axes({ length: 2.2, labels: true });
    g.lineThrough([0, 0, 0], AX3, { color: 'gold', width: 2.6, len: 3.2 });
    g.box(R, { fill: 'v', alpha: 0.07, stroke: 'v', strokeAlpha: 0.7 });
    g.vector([0, 0, 0], N.scale(AX3, 2.2), { color: 'gold', width: 3.4, label: 'a', labelFont: 'math' });
    
    [[1, 0, 0], [0, 0, 1], [0, 1, 1]].forEach(c => {
      const tr = []; for (let i = 0; i <= 24; i++) tr.push(N.matVec(N.rot3(AX3, s.theta3 * i / 24), c));
      g.polyline(tr, { color: 'hud', width: 1.4, dashed: true, alpha: 0.75 });
      g.point(N.matVec(R, c), { r: 4.5, color: 'white' });
    });
    g.point([0, 0, 0], { r: 3, color: 'gold' });
    g.hud('Only in 3D does every rotation have an axis.', 14, 24, { color: 'gold', size: 15, bold: true });
    g.hud('rotation angle ' + LA.fmt.deg(s.theta3) + ' about the gold axis a', 14, 46, { color: 'hud', size: 12.5 });
    g.hud('a: eigenvector with λ = 1 — points on this line do not move', 14, 66, { color: 'gold', size: 12.5 });
  }

  /* ---------------- ML tie-in: three thumbnails ---------------- */
  function renderML(g, s) {
    const u = s.mlT;
    // 1. covariance cloud with gold principal axes
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 2.6 }, border: true }, (g2, r) => {
      g2.ghostGrid();
      g2.points(s.cloud, { r: 3, color: 'data', alpha: 0.85 });
      const a = seg(u, 0, 0.35);
      s.pcVec.forEach((q, i) => {
        const L = 2 * Math.sqrt(s.pcVal[i]) * a;
        g2.lineThrough([0, 0], q, { color: 'gold', width: 1.3, alpha: 0.5 * a, len: 4.5 });
        if (L > 0.05) g2.vector(N.scale(q, L), { color: 'gold', width: 2.6, head: 10, label: i ? 'q₂' : 'q₁', labelFont: 'math' });
      });
      g2.hud('covariance Σ: its eigenvectors are the principal components (p4-10)', r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
      g2.hud('60 synthetic points (LA.rng seed 49)', r.x + 12, r.y + r.h - 14, { size: 11.5, color: 'muted' });
    });
    // 2. a bowl's elliptical contours with gold axes and the 2/λmax step-size limit
    const H = [[3, 1], [1, 1.2]], eh = N.eigSym(H), lmax = eh.values[0];
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [1.2, -0.2], halfH: 2.5 }, border: true }, (g2, r) => {
      const f = p => 0.5 * (H[0][0] * p[0] * p[0] + 2 * H[0][1] * p[0] * p[1] + H[1][1] * p[1] * p[1]);
      g2.contours(f, [0.15, 0.5, 1.1, 2], { x0: -2.6, x1: 2.6, y0: -2.4, y1: 2.4 }, { color: 'teal', width: 1.4, res: 40 });
      eh.vectors.forEach((q, i) => g2.vector(N.scale(q, i ? 1.5 : 0.9), { color: 'gold', width: 2.4, head: 9, label: i ? 'q₂' : 'q₁', labelFont: 'math' }));
      // learning-rate ruler: η from 0 to 1, red tick at 2/λmax
      const x0 = 2.9, x1 = 6.2, yy = -1.2, X = e => x0 + (x1 - x0) * e / 1.0, lim = 2 / lmax;
      g2.line([x0, yy], [x1, yy], { color: 'hud', width: 2 });
      g2.line([X(lim), yy - 0.25], [X(lim), yy + 0.25], { color: 'bad', width: 2.4 });
      g2.label('2/λmax = ' + LA.fmt(lim), [X(lim), yy + 0.25], { color: 'bad', size: 12, anchor: 'above', font: 'ui' });
      const eta = 0.05 + 0.9 * u;
      g2.point([X(eta), yy], { r: 6, color: eta < lim ? 'ok' : 'bad', label: 'η', labelColor: 'hud', dy: 16 });
      g2.label(eta < lim ? 'safe step' : 'diverges', [X(eta), yy], { color: eta < lim ? 'ok' : 'bad', size: 12, font: 'ui', dy: 34 });
      g2.hud('Hessian H: the largest eigenvalue caps the learning rate (p5-03)', r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
    });
    // 3. unit circle under Wᵏ
    const Q = N.rot2(Math.PI / 6), lw = [1.15, 0.72];
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 2.5 }, border: true }, (g2, r) => {
      const k = 6 * u;
      const Wk = N.matMul(N.matMul(Q, N.diag([Math.pow(lw[0], k), Math.pow(lw[1], k)])), N.transpose(Q));
      g2.lineThrough([0, 0], N.col(Q, 0), { color: 'gold', width: 1.4, alpha: 0.7 });
      g2.lineThrough([0, 0], N.col(Q, 1), { color: 'gold', width: 1.4, alpha: 0.7 });
      g2.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1 });
      for (let j = 1; j < Math.floor(k + 1e-9) + 1 && j <= 6; j++) { const Wj = N.matMul(N.matMul(Q, N.diag([Math.pow(lw[0], j), Math.pow(lw[1], j)])), N.transpose(Q)); g2.ellipse(Wj, [0, 0], { stroke: 'w', width: 1, strokeAlpha: 0.3 }); }
      g2.ellipse(Wk, [0, 0], { stroke: 'w', width: 2.4, fill: 'w', alpha: 0.12 });
      g2.label('λ = 1.15: grows', N.scale(N.col(Q, 0), 2.6), { color: 'gold', size: 12, font: 'ui' });
      g2.label('λ = 0.72: vanishes', N.scale(N.col(Q, 1), 1.5), { color: 'gold', size: 12, font: 'ui', dx: -40 });
      g2.hud('weights Wᵏ: |λ| > 1 explodes, |λ| < 1 vanishes (p5-06)   k = ' + LA.fmt.fixed(k, 1), r.x + 12, r.y + 18, { size: 12.5, color: 'hud', bold: true });
    });
  }

  /* ---------------- recap cards ---------------- */
  function renderRecap(g, s) {
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0.9, 0], halfH: 2.4 }, border: true }, (g2, r) => {
      g2.ghostGrid();
      g2.lineThrough([0, 0], [1, 0], { color: 'gold', width: 2, alpha: 0.85 });
      g2.lineThrough([0, 0], [1, -1], { color: 'gold', width: 2, alpha: 0.85 });
      FAN.forEach(u => { const on = onLine(u, [1, 0]) || onLine(u, [1, -1]); g2.vector(N.scale(N.matVec(A0, u), 0.62), { color: on ? 'gold' : 'muted', width: on ? 2.2 : 1.2, head: on ? 8 : 6, alpha: on ? 1 : 0.7 }); });
      g2.label('×3', [1.86, 0], { color: 'gold', size: 13, font: 'ui', bold: true, dy: -14 });
      g2.label('×2', [0.88, -0.88], { color: 'gold', size: 13, font: 'ui', bold: true, dx: 18 });
      g2.hud('Keeps its line: Av = λv', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [2.25, 0.7], halfH: 1.75 }, border: true }, (g2, r) => {
      g2.line([-0.3, 0], [4.8, 0], { color: 'gridAxis', width: 1.4 });
      const pts = []; for (let i = 0; i <= 60; i++) { const l = -0.2 + 4.9 * i / 60; pts.push([l, 0.45 * detAt(A0, l)]); }
      g2.polyline(pts, { color: 'hud', width: 2.2 });
      [2, 3].forEach(l => { g2.point([l, 0], { r: 6, color: 'gold' }); g2.label('λ = ' + l, [l, 0], { color: 'gold', size: 13, font: 'ui', dy: 20 }); });
      g2.label('det(A − λI)', [4.2, 0.45 * detAt(A0, 4.2)], { color: 'hud', size: 12.5, font: 'ui', anchor: 'left', dx: 12 });
      g2.hud('Collapse where det(A − λI) = 0', r.x + 12, r.y + 18, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 0.5, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 2.3 }, border: true }, (g2, r) => {
      g2.circle([0, 0], 1, { stroke: 'residual', dashed: true, width: 1 });
      g2.ellipse(SYM, [0, 0], { stroke: 'teal', width: 2, fill: 'teal', alpha: 0.08 });
      const q1 = N.normalize([1, 1]), q2 = N.normalize([1, -1]);
      g2.vector(N.scale(q1, 3), { color: 'gold', width: 2.4, head: 9, label: 'q₁', labelFont: 'math' });
      g2.vector(q2, { color: 'gold', width: 2.4, head: 9, label: 'q₂', labelFont: 'math' });
      g2.arc([0, 0], q1, q2, { r: 0.35, color: 'gold', right: true });
      g2.hud('Symmetric: ⟂ axes', r.x + 10, r.y + 18, { size: 12.5, color: 'hud', bold: true });
    });
    g.viewport({ x: 0.5, y: 2 / 3, w: 0.5, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 1.6 }, border: true }, (g2, r) => {
      // every 60°: a quarter-turn moves each arrow off all the dashed originals
      FAN.filter((u, i) => i % 4 === 0).forEach(u => { g2.vector(u, { color: 'residual', dashed: true, width: 1, head: 5, alpha: 0.5 }); g2.vector(N.matVec(ROT, u), { color: 'muted', width: 1.4, head: 6 }); });
      g2.marker.cross([0, 0], { size: 26 });
      g2.hud('Rotation: none in 2D', r.x + 10, r.y + 18, { size: 12.5, color: 'hud', bold: true });
      g2.hud('(an axis only in 3D)', r.x + 10, r.y + r.h - 14, { size: 11.5, color: 'muted' });
    });
  }
})();
