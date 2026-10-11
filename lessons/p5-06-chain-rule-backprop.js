/* p5-06 — Backprop as a chain of Jacobians; outer-product gradients. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, lerpM = LA.lerpM, C = LA.color;

  /* ---------- the network (storyboard weights) ---------- */
  const X0 = [0.8, -0.4];
  const W1_0 = [[1.2, -0.6], [0.5, 1.0]], B1 = [0.1, -0.2];
  const W2_0 = [[0.8, 0.9], [-1.1, 0.4]], B2 = [0, 0.1];
  const W3 = [1.0, -0.8], B3 = 0, ETA = 0.3;
  /* ---------- layout (world units; local panel coordinates = world offsets) ----------
     U-shaped flow: x (top-left) → h₁ (top-right) → h₂ (bottom-right) → output y below it; the bottom-left
     quadrant (x −2.75…−0.1, y −2.3…0.3) holds tiles, heatmaps and insets. */
  const PX = [-1.38, 1.38, 1.38], PY = [1.72, 1.72, -1.0], PR = 1.25, E = 0.5, GS = 0.5;
  const OX = 1.38, OY = -2.62, OS = 0.5, E0 = E;
  const NET_CAM = { center: [0, 0], halfH: 3.32 };
  const WIDE_CAM = { center: [0, 0.1], halfH: 4.95 };
  const DX = [-2.6, 0, 2.6], DY = [3.45, 1.05, -1.35], DH = 1.1, DU = 0.8, TY = -3.0;
  const SUBS = '₀₁₂₃₄₅₆₇₈₉', SUPS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const sub = k => String(k).split('').map(ch => SUBS[+ch]).join('');
  const sup = k => String(k).split('').map(ch => SUPS[+ch]).join('');
  const P = (k, v) => [PX[k] + v[0], PY[k] + v[1]];
  const wideCam = () => Object.assign(LA.cam.defaults2d(), LA.clone(WIDE_CAM));
  const clampL = (v, r) => [LA.clamp(v[0], -r, r), LA.clamp(v[1], -r, r)];
  const divColor = v => C.diverge(LA.clamp(v, -1, 1));
  const divRamp = t => C.diverge(t * 2 - 1);
  /** static number for the math panel (tex.num mis-aligns when a panel also holds matrices; see report) */
  const nT = (x, color, digits) => '<span class="tx-nv ' + LA.tex.colorClass(color) + '">' + LA.fmt(x, digits) + '</span>';

  /* ---------- pure numerics ---------- */
  function actF(a, z) { return a === 'relu' ? Math.max(0, z) : a === 'identity' ? z : Math.tanh(z); }
  function actD(a, z) { if (a === 'relu') return z > 0 ? 1 : 0; if (a === 'identity') return 1; const t = Math.tanh(z); return 1 - t * t; }
  const dJ = (D, W) => [[D[0] * W[0][0], D[0] * W[0][1]], [D[1] * W[1][0], D[1] * W[1][1]]];
  /** forward and backward pass at input x (default s.x); weights are scaled by s.c */
  function fw(s, xIn) {
    const x = xIn || s.x, a = s.act, c = s.c;
    const W1 = N.scaleM(s.W1, c), W2 = N.scaleM(s.W2, c);
    const z1 = N.add(N.matVec(W1, x), B1), h1 = z1.map(z => actF(a, z)), D1 = z1.map(z => actD(a, z));
    const z2 = N.add(N.matVec(W2, h1), B2), h2 = z2.map(z => actF(a, z)), D2 = z2.map(z => actD(a, z));
    const y = N.dot(W3, h2) + B3, L = 0.5 * (y - s.tgt) * (y - s.tgt);
    const g3 = y - s.tgt, gh2 = N.scale(W3, g3), d2 = [D2[0] * gh2[0], D2[1] * gh2[1]];
    const gh1 = N.matVec(N.transpose(W2), d2), d1 = [D1[0] * gh1[0], D1[1] * gh1[1]];
    const gx = N.matVec(N.transpose(W1), d1);
    const J1 = dJ(D1, W1), J2 = dJ(D2, W2);
    return { x, W1, W2, z1, h1, D1, z2, h2, D2, y, L, g3, gh2, d2, gh1, d1, gx, J1, J2, J21: N.matMul(J2, J1), dW1: N.outer(d1, x), dW2: N.outer(d2, h1) };
  }
  /** deep stack: depth layers W = c·R(30°); circle images, a tracked point and pulled-back gradients */
  function deepRun(s) {
    const d = s.depth, a = s.act, W = N.scaleM(N.rot2(Math.PI / 6), s.c), WT = N.transpose(W);
    const circ = []; for (let i = 0; i < 48; i++) { const t = i / 48 * Math.PI * 2; circ.push([Math.cos(t), Math.sin(t)]); }
    const circles = [circ], pts = [[0.8, 0.6]], Ds = [null];
    for (let k = 1; k <= d; k++) {
      circles.push(circles[k - 1].map(p => N.matVec(W, p).map(z => actF(a, z))));
      const z = N.matVec(W, pts[k - 1]); pts.push(z.map(v => actF(a, v))); Ds.push(z.map(v => actD(a, v)));
    }
    const gs = new Array(d + 1); gs[d] = [-0.6, 0.8];
    for (let k = d; k >= 1; k--) gs[k - 1] = N.matVec(WT, [Ds[k][0] * gs[k][0], Ds[k][1] * gs[k][1]]);
    return { circles, pts, gs, norms: gs.map(v => N.norm(v)) };
  }
  let _batch = null;
  /** four synthetic samples δ ∈ ℝ⁶, x ∈ ℝ⁸ (seed 506), their rank-1 sheets and running sums */
  function batchData() {
    if (_batch) return _batch;
    const rand = LA.rng(506), sheets = [], sums = [], ranks = [];
    for (let n = 0; n < 4; n++) { const d = LA.rng.gaussV(rand, 6), x = LA.rng.gaussV(rand, 8); sheets.push(N.outer(d, x)); }
    let acc = N.zerosM(6, 8);
    for (let n = 0; n < 4; n++) { acc = N.addM(acc, sheets[n]); sums.push(acc); ranks.push(N.rank(acc)); }
    const r1 = Math.max(...sheets.map(M => Math.max(...M.map(r => Math.max(...r.map(Math.abs))))));
    const r2 = Math.max(...acc.map(r => Math.max(...r.map(Math.abs))));
    const rand2 = LA.rng(507); const thumb = N.outer(LA.rng.gaussV(rand2, 12), LA.rng.gaussV(rand2, 12));
    _batch = { sheets, sums, ranks, r1, r2, thumb };
    return _batch;
  }

  function resetAll(s) {
    Object.assign(s, {
      scene: 'net', x: X0.slice(), c: 1, act: 'tanh', W1: LA.clone(W1_0), W2: LA.clone(W2_0), tgt: 0, depth: 8,
      fwd: 3, sqA: 0, dBars: false, hop: -1, prodA: 0, tiles: 0, gxStrip: false, back: 0, tdot: 0, outerA: 0, collapse: 0,
      batchU: 0, satInset: false, deepF: 8, deepB: 8, cross: false, gd: 0, mlTie: false, recap: false, showOuter: false,
      guessOuter: [[1, 0], [0, 1]], mathMode: 'fwd',
    });
    s.cam = Object.assign(LA.cam.defaults2d(), LA.clone(NET_CAM));
  }

  LA.registerLesson({
    id: 'p5-06-chain-rule-backprop',
    part: 5,
    title: 'Backprop as a chain of Jacobians; outer-product gradients',
    subtitle: 'Forward, Jacobians multiply; backward, transposes pull the loss arrow home; weight gradients are rank 1.',
    minutes: 14,
    prereqs: ['p5-05-jacobian', 'p2-04-composition', 'p2-10-transpose-outer-symmetric', 'p2-12-neural-layer', 'p5-03-gradient-descent', 'p4-03-diagonalization-powers'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D layers (width 2)', realSize: 'GPT-scale layers have W of 4096×4096 and 96 chained layers; ∂L/∂W per token is still rank 1', transfers: 'chain rule = Jacobian product, backward pass = Jᵀ, ∂L/∂W = δxᵀ, norms multiply layer by layer — exact for any widths and depths' },
    cantShow: ['Jacobians of size 10³×10³; only their 2×2 analogues', 'ReLU’s piecewise Jacobians in the main story (we use tanh); skip connections get one sentence', 'the batch dimension beyond the four-sample heatmap sum'],
    terms: [
      { term: 'chain rule', def: 'The derivative of a chain of maps is the product of their Jacobians, outermost first.' },
      { term: 'backprop', def: 'Computing every gradient by pulling the loss gradient backward through each layer with the transposed Jacobian.' },
      { term: 'delta', def: 'δ: the gradient of the loss with respect to a layer’s pre-activation z.' },
      { term: 'outer product', def: 'δxᵀ: the rank-1 matrix whose (i, j) entry is δᵢ·xⱼ.' },
      { term: 'vanishing gradient', def: 'The pulled-back gradient shrinking geometrically as it passes through more layers.' },
      { term: 'exploding gradient', def: 'The pulled-back gradient growing geometrically as it passes through more layers.' },
    ],
    atlas: ['backprop', 'mlp', 'optimization'],
    thumbnail: 4,

    state() {
      const s = { view: '2d', cam: LA.cam.defaults2d() };
      resetAll(s);
      return s;
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g, s); return; }
      if (s.scene === 'batch') { renderBatch(g, s); return; }
      if (s.scene === 'deep') { renderDeep(g, s, ctx); return; }
      if (s.scene === 'outer') { renderOuterCheck(g, s); return; }
      renderNet(g, s, ctx);
    },

    math(s, tex) {
      const m = s.mathMode, sym = (a, o) => tex.sym(a, o), op = tex.op, eq = tex.eq;
      if (m === 'none') return '';
      if (s.scene === 'deep' || m === 'deep' || m === 'norms') {
        const r = deepRun(s), d = s.depth, ratio = r.norms[0] / r.norms[d];
        const l1 = eq(sym('‖g₀‖'), op('='), sym('c'), sup(d), sym('‖g' + sub(d) + '‖'), op('='), nT(s.c, 'gold'), sup(d), sym('‖g' + sub(d) + '‖'), op('≈'), nT(Math.pow(s.c, d), null, 3), sym('‖g' + sub(d) + '‖'), tex.text(s.act === 'identity' ? '(identity)' : '(identity only)'));
        const l2 = eq(tex.text('measured ' + s.act + ':'), sym('‖g₀‖ / ‖g' + sub(d) + '‖'), op('='), nT(ratio, ratio < 0.5 || ratio > 2 ? 'bad' : 'ok', 3));
        if (m === 'norms') return tex.lines(eq(sym('‖gₖ‖'), op('≈'), sym('c'), '<sup>' + d + '−k</sup>', sym('‖g' + sub(d) + '‖'), op('·'), tex.text('orthogonal W:'), sym('‖Wᵀg‖ = ‖g‖')), l2);
        return tex.lines(l1, l2);
      }
      if (s.scene === 'batch' || m === 'batch') { const B = batchData(), k = Math.max(1, Math.min(4, Math.round(s.batchU * 4))); return tex.lines(eq(sym('∂L/∂W'), op('='), tex.text('Σₙ'), sym('δ⁽ⁿ⁾x⁽ⁿ⁾ᵀ')), eq(tex.text('rank ≤ batch size · rank of the sum of ' + k + ':'), nT(B.ranks[k - 1], 'gold'))); }
      if (m === 'guess') return tex.lines(eq(sym('δ', { color: 'v' }), op('='), tex.vec([2, -1], { color: 'v' }), op('·'), sym('xᵀ'), op('='), tex.rowvec([3, 1])), eq(tex.text('yours:'), tex.mat(s.guessOuter, { colors: 'w' })));
      const f = fw(s);
      if (m === 'fwd') return tex.lines(
        eq(sym('h₁'), op('='), tex.text(s.act), tex.paren(sym('W₁x') + op('+') + sym('b₁')), op('='), tex.vec(f.h1)),
        eq(sym('h₂'), op('='), tex.text(s.act), tex.paren(sym('W₂h₁') + op('+') + sym('b₂')), op('='), tex.vec(f.h2)),
        eq(sym('y'), op('='), sym('w₃·h₂'), op('+'), sym('b₃'), op('='), nT(f.y, 'v'), op('·'), sym('L'), op('='), tex.text('½(y − t)² ='), nT(f.L, 'gold')));
      if (m === 'J') return tex.lines(
        eq(sym('J₁'), op('='), sym('D₁W₁'), op('='), tex.mat(f.J1, { colors: 'cols' }), op('·'), sym('J₂'), op('='), sym('D₂W₂'), op('='), tex.mat(f.J2, { colors: 'cols' })),
        eq(sym('Dₗ'), op('='), tex.text('diag(1 − tanh²(zₗ)) :'), sym('D₁'), op('='), tex.text('diag('), nT(f.D1[0], 'i'), op(','), nT(f.D1[1], 'j'), tex.text(')'), sym('D₂'), op('='), tex.text('diag('), nT(f.D2[0], 'i'), op(','), nT(f.D2[1], 'j'), tex.text(')')));
      if (m === 'chain') return tex.lines(eq(sym('∂h₂/∂x'), op('='), sym('J₂'), sym('J₁'), op('='), tex.mat(f.J21, { colors: 'cols' })), eq(tex.text('composition = multiplication, right to left (p2-04)')));
      if (m === 'loss') return tex.lines(eq(sym('∂L/∂x'), op('='), tex.paren(sym('y') + op('−') + sym('t')), sym('w₃ᵀ'), sym('J₂'), sym('J₁'), op('='), tex.rowvec(f.gx, { color: 'v' })), eq(tex.text('a 1×2 row: (y − t) ='), nT(f.g3, 'v')));
      if (m === 'back') return tex.lines(
        eq(sym('gₗ₋₁'), op('='), sym('Jₗᵀgₗ'), op('='), sym('Wₗᵀ'), tex.paren(sym('Dₗgₗ')), op('·'), sym('δₗ', { color: 'v' }), op('='), sym('Dₗgₗ')),
        eq(sym('δ₂', { color: 'v' }), op('='), tex.vec(f.d2, { color: 'v' }), sym('∂L/∂h₁'), op('='), tex.vec(f.gh1, { color: 'v' }), sym('δ₁', { color: 'v' }), op('='), tex.vec(f.d1, { color: 'v' }), sym('∂L/∂x'), op('='), tex.vec(f.gx, { color: 'v' })));
      if (m === 'dot') { const d = [0.3, 0.1], Jd = N.matVec(f.J1, d); return tex.lines(eq(tex.paren(sym('J', {}) + sym('d', { color: 'w' })), op('·'), sym('g', { color: 'v' }), op('='), sym('d', { color: 'w' }), op('·'), tex.paren(sym('Jᵀ') + sym('g', { color: 'v' }))), eq(nT(N.dot(Jd, f.gh1), 'gold', 3), op('='), nT(N.dot(d, f.gx), 'gold', 3))); }
      if (m === 'outer') return tex.lines(eq(sym('∂L/∂W₂'), op('='), sym('δ₂', { color: 'v' }), sym('h₁ᵀ'), op('='), tex.mat(f.dW2, { colors: (i, j) => j === 0 ? 'i' : 'j' })), eq(tex.text('(∂L/∂W₂)ᵢⱼ = δ₂ᵢ · h₁ⱼ · rank 1')));
      if (m === 'sat') return tex.lines(
        eq(sym('c'), op('='), nT(s.c, 'gold'), op('·'), sym('z₁'), op('='), tex.vec(f.z1), op('·'), sym('D₁'), op('='), tex.text('diag('), nT(f.D1[0], 'i', 4), op(','), nT(f.D1[1], 'j'), tex.text(')')),
        eq(sym('δ₁', { color: 'v' }), op('='), tex.vec(f.d1, { color: 'v', digits: 3 }), op('·'), sym('‖∂L/∂x‖'), op('='), nT(N.norm(f.gx), 'v')));
      if (m === 'gd') return tex.lines(eq(sym('W₂'), op('←'), sym('W₂'), op('−'), sym('η'), sym('δ₂', { color: 'v' }), sym('h₁ᵀ'), op('·'), sym('η'), op('='), nT(ETA)), eq(sym('L'), op(':'), nT(s._L0 === undefined ? f.L : s._L0, 'muted'), op('→'), nT(f.L, 'gold')));
      // playground
      return tex.lines(eq(sym('L'), op('='), nT(f.L, 'gold'), op('·'), sym('‖∂L/∂x‖'), op('='), nT(N.norm(f.gx), 'v'), op('·'), sym('c'), op('='), nT(s.c)),
        eq(sym('∂L/∂W₁'), op('='), sym('δ₁x', {}), tex.text('ᵀ ='), tex.mat(f.dW1, { colors: (i, j) => j === 0 ? 'i' : 'j' }), sym('∂L/∂W₂'), op('='), tex.mat(f.dW2, { colors: (i, j) => j === 0 ? 'i' : 'j' })));
    },

    steps: [
      { title: 'The network from p2-12, as three spaces', duration: 4,
        narration: 'Two layers and a readout: h₁ = tanh(W₁x + b₁), h₂ = tanh(W₂h₁ + b₂), y = w₃·h₂ + b₃, with loss L = ½(y − t)² and target t = 0. Watch the yellow input <span class="c-v">x</span> travel left to right through three spaces, each drawn with its own grid.',
        setup(s) { resetAll(s); s.fwd = 0; s.mathMode = 'fwd'; },
        at(s, t) { s.fwd = seg(t, 0.2, 0.5) + seg(t, 0.5, 0.8) + seg(t, 0.8, 1); } },

      { title: 'Each layer is locally a matrix', duration: 3,
        narration: 'From p5-05: around the current point each layer acts like its Jacobian Jₗ = DₗWₗ. A tiny square at x, with a green and a red side, becomes a parallelogram at h₁; a tiny square at h₁ becomes one at h₂. The bars below show the tanh slopes in D₁ and D₂.',
        setup(s) { s.fwd = 3; s.sqA = 0; s.dBars = true; s.mathMode = 'J'; },
        at(s, t) { s.sqA = seg(t, 0.1, 0.6); } },

      { title: 'Compose: the product of Jacobians', duration: 4,
        narration: 'Send the square at x through both layers, one hop at a time. The result equals sending it through the single matrix J₂J₁, drawn dashed purple on top: composition is multiplication, right to left (p2-04). That is the <b>chain rule</b>: the derivatives of a chain multiply.',
        setup(s) { s.sqA = 0; s.dBars = false; s.hop = 0; s.prodA = 0; s.mathMode = 'chain'; },
        at(s, t) { s.hop = 2 * seg(t, 0, 0.6, 'linear'); s.prodA = seg(t, 0.62, 0.9); } },

      { title: 'The whole chain to the loss', duration: 3,
        narration: 'Add the readout and the loss: ∂L/∂x = (y − t)·w₃ᵀ·J₂·J₁, a 1×2 row of slopes, shown under the input. The tiles sit above their own spaces, so the picture runs left to right while the formula reads right to left. Four small matrices, one product.',
        setup(s) { s.hop = -1; s.prodA = 0; s.tiles = 0; s.gxStrip = false; s.mathMode = 'loss'; },
        at(s, t) { s.tiles = 4 * seg(t, 0.05, 0.8, 'linear'); s.gxStrip = t > 0.82; } },

      { title: 'Backward = transpose pull-back', duration: 5,
        narration: 'Now compute from the loss backward: <b>backprop</b>. Each stage’s gradient is the transpose of the next Jacobian times the gradient after it (p2-10). The yellow arrow ∂L/∂h₂ shrinks through D₂ into the <b>delta</b> δ₂, then W₂ᵀ carries it to h₁, then D₁ and W₁ᵀ carry it home to x.',
        setup(s) { s.tiles = 0; s.gxStrip = false; s.back = 0; s.mathMode = 'back'; },
        at(s, t) { s.back = seg(t, 0, 1, 'linear'); } },

      { title: 'Why the transpose', duration: 4,
        narration: 'J moves a step <span class="c-w">d</span> forward; Jᵀ moves a gradient <span class="c-v">g</span> backward so that the slopes match: (Jd)·g = d·(Jᵀg). The slope seen after the layer equals the slope of the pulled-back arrow before it. The two bars are equal.',
        setup(s) { s.back = 0; s.tdot = 0; s.mathMode = 'dot'; },
        at(s, t) { s.tdot = seg(t, 0.1, 0.6); } },

      { title: 'Weight gradients are outer products', duration: 5,
        narration: 'What about the weights? For layer 2, ∂L/∂W₂ = δ₂h₁ᵀ: the yellow column δ₂ times the row h₁ᵀ, an <b>outer product</b> (p2-10). Cell (i, j) is δ₂ᵢ·h₁ⱼ. Every column is a multiple of δ₂, so the matrix is rank 1: it squashes the plane onto the gold line of δ₂.',
        setup(s) { s.tdot = 0; s.outerA = 0; s.collapse = 0; s.mathMode = 'outer'; },
        at(s, t) { s.outerA = seg(t, 0, 0.6, 'linear'); s.collapse = seg(t, 0.62, 1); } },

      { title: 'A batch sums rank-1 sheets', duration: 4,
        narration: 'Four samples give four rank-1 sheets; the batch gradient is their sum, of rank at most four. Shown for a synthetic 6×8 weight matrix: one sample is stripes, every row a multiple of one row; four samples blend. The picture stops at 3. The formula does not.',
        setup(s) { s.outerA = 0; s.collapse = 0; s.scene = 'batch'; s.batchU = 0; s.mathMode = 'batch'; s.cam = wideCam(); },
        at(s, t) { s.batchU = seg(t, 0, 0.75, 'linear'); } },

      { title: 'Saturation kills the pull-back', duration: 4,
        narration: 'Multiply the weights by 4. Now z₁ = (4.9, −0.2): the first tanh unit sits on its flat shoulder, slope 0.0002. Layer 1’s parallelogram collapses to a sliver and δ₁ loses its first component, so everything reaching x flows through the second unit alone. ‖∂L/∂x‖ stays near 0.75 here; saturate both units (check 2) and it vanishes.',
        setup(s) { s.scene = 'net'; s.batchU = 0; s.c = 1; s.sqA = 1; s.back = 1; s.satInset = true; s.mathMode = 'sat'; s.cam = Object.assign(LA.cam.defaults2d(), LA.clone(NET_CAM)); },
        at(s, t) { s.c = LA.lerp(1, 4, seg(t, 0.1, 0.8)); } },

      { title: '✕ “Sizes stay about the same through many layers”', duration: 4, misconception: true,
        narration: 'A common belief: “sizes stay about the same through many layers.” Stack eight identical layers with W = 0.6·R(30°). The unit circle shrinks to radius 0.6⁸ = 0.017, and the gradient pulled back through eight transposes shrinks the same way (p4-03 powers). Sizes do not stay put; they multiply.',
        setup(s) { s.scene = 'deep'; s.sqA = 0; s.back = 0; s.satInset = false; s.c = 0.6; s.act = 'identity'; s.depth = 8; s.deepF = 0; s.deepB = 0; s.cross = false; s.mathMode = 'deep'; s.cam = wideCam(); },
        at(s, t) { s.deepF = 8 * seg(t, 0, 0.45, 'linear'); s.deepB = 8 * seg(t, 0.45, 0.85, 'linear'); s.cross = t > 0.86; } },

      { title: 'Per-layer gradient norms', duration: 5,
        narration: 'Read the bars: ‖gₖ‖ at every layer. Scale 0.6: they decay, a <b>vanishing gradient</b>. Scale 1.6: an <b>exploding gradient</b>. Scale 1, a pure rotation — an orthogonal W (p3-04) — keeps every bar level, because rotations preserve length. Switch tanh on and even the orthogonal case leaks, since every tanh slope is below 1.',
        setup(s) { s.cross = false; s.deepF = 8; s.deepB = 8; s.act = 'identity'; s.c = 0.6; s.mathMode = 'norms'; },
        at(s, t) {
          s.c = t < 0.25 ? 0.6 : t < 0.3 ? LA.lerp(0.6, 1.6, seg(t, 0.25, 0.3)) : t < 0.5 ? 1.6 : t < 0.55 ? LA.lerp(1.6, 1, seg(t, 0.5, 0.55)) : 1;
          s.act = t > 0.78 ? 'tanh' : 'identity';
        } },

      { title: 'Why ML cares: this is one training step', duration: 4, mlTie: true,
        narration: 'Every framework computes exactly these Jᵀ products and outer products, then steps each layer: W ← W − η·δxᵀ. Watch one step with η = 0.3: the loss drops from 0.56 to 0.28. A 4096×4096 layer gets one rank-1 update per sample, summed over the batch; p6-08 trains a real classifier this way.',
        setup(s) { resetAll(s); s.mlTie = true; s.mathMode = 'gd'; const f = fw(s); s._L0 = f.L; s._W1 = LA.clone(s.W1); s._W2 = LA.clone(s.W2); s._dW1 = f.dW1; s._dW2 = f.dW2; s.gd = 0; },
        at(s, t) { s.gd = seg(t, 0.15, 0.75); s.W1 = N.subM(s._W1, N.scaleM(s._dW1, ETA * s.gd)); s.W2 = N.subM(s._W2, N.scaleM(s._dW2, ETA * s.gd)); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Forward, each layer is a Jacobian and the chain rule multiplies them. Backward, the loss arrow is pulled back by Jᵀ, shrinking where tanh is flat. Weight gradients are rank-1 outer products δxᵀ, and norms multiply layer by layer — level only for near-orthogonal weights.',
        cards: ['<b>Chain rule = product of Jacobians.</b> Two parallelograms chained equal the one of J₂J₁.', '<b>Backward = Jᵀ pull-back.</b> The loss arrow shrinks where tanh saturates.', '<b>∂L/∂W = δxᵀ, rank 1.</b> Norms multiply with depth: c⁸ over eight layers.'],
        setup(s) { resetAll(s); s.recap = true; s.mathMode = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the yellow input <span class="c-v">x</span> (network view) or the gold weight-scale knob (deep view). Every forward value, Jacobian, pulled-back arrow and outer product is live. The GD button takes one step W ← W − η·δxᵀ on both layers.</p>',
      setup(s) { resetAll(s); s.sqA = 1; s.back = 1; s.showOuter = true; s.mathMode = 'play'; },
      controls: [
        { type: 'preset', label: 'Preset', items: [
          { label: 'Healthy', apply(s) { resetAll(s); s.sqA = 1; s.back = 1; s.showOuter = true; s.mathMode = 'play'; } },
          { label: 'Saturated (×4)', apply(s) { resetAll(s); s.c = 4; s.sqA = 1; s.back = 1; s.showOuter = true; s.mathMode = 'play'; } },
          { label: 'Deep: vanish', apply(s) { s.scene = 'deep'; s.c = 0.6; s.act = 'identity'; s.depth = 8; s.mathMode = 'norms'; s.cam = wideCam(); } },
          { label: 'Deep: orthogonal', apply(s) { s.scene = 'deep'; s.c = 1; s.act = 'identity'; s.depth = 8; s.mathMode = 'norms'; s.cam = wideCam(); } },
          { label: 'Deep: explode', apply(s) { s.scene = 'deep'; s.c = 1.6; s.act = 'identity'; s.depth = 8; s.mathMode = 'norms'; s.cam = wideCam(); } },
        ] },
        { type: 'slider', bind: 'c', min: 0.2, max: 4, step: 0.05, label: 'Weight scale c', color: 'gold' },
        { type: 'slider', bind: 'depth', min: 2, max: 8, step: 1, label: 'Depth (deep view)' },
        { type: 'preset', label: 'Activation', items: [
          { label: 'tanh', values: { act: 'tanh' } },
          { label: 'ReLU', values: { act: 'relu' } },
          { label: 'identity', values: { act: 'identity' } },
        ] },
        { type: 'toggle', bind: 'showOuter', label: 'Outer-product heatmaps (network view)' },
        { type: 'button', label: 'One GD step (η = 0.3)', onClick(s) { if (s.scene !== 'net') return; const f = fw(s); s.W1 = N.subM(s.W1, N.scaleM(f.dW1, ETA * s.c)); s.W2 = N.subM(s.W2, N.scaleM(f.dW2, ETA * s.c)); } },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the gold knob to set the weight scale c so that, through 8 identity layers, the gradient norm at the input stays between one half and twice its value at the output.',
        setup(s) { resetAll(s); s.scene = 'deep'; s.act = 'identity'; s.depth = 8; s.c = 0.4; s.mathMode = 'norms'; s.cam = wideCam(); },
        judge(s) { const r = deepRun(s), q = r.norms[0] / r.norms[s.depth]; return { ok: q >= 0.5 && q <= 2, progress: Math.max(0, 1 - Math.abs(Math.log2(q)) / 6) }; },
        hint: 'The ratio is c⁸; you need c very close to 1.',
        solution(s) { s.c = 1; } },
      { kind: 'manipulate',
        prompt: 'The weights are scaled by 4. Drag the yellow input <span class="c-v">x</span> into a region where the first layer saturates so that the pulled-back gradient at x has norm below 0.05.',
        setup(s) { resetAll(s); s.c = 4; s.x = [0.1, 0.1]; s.sqA = 1; s.back = 1; s.satInset = true; s.mathMode = 'sat'; },
        judge(s) { const n = N.norm(fw(s).gx); return { ok: n < 0.05, progress: Math.max(0, Math.min(1, 1 - (n - 0.05) / 0.8)) }; },
        hint: 'Push x where both z₁ components are large — far from the lines W₁x + b₁ = 0. Try a corner of the input panel.',
        solution(s) { let best = s.x, bv = Infinity; for (let i = -24; i <= 24; i++) for (let j = -24; j <= 24; j++) { const x = [i * 0.05, j * 0.05], n = N.norm(fw(s, x).gx); if (n < bv) { bv = n; best = x; } } s.x = best; } },
      { kind: 'entry',
        prompt: 'Type the outer product δxᵀ for δ = (2, −1) and x = (3, 1).',
        bind: 'guessOuter', answer: [[6, 2], [-3, -1]], tol: LA.TOL.entry, label: 'δxᵀ', colors: 'cols',
        setup(s) { resetAll(s); s.scene = 'outer'; s.guessOuter = [[1, 0], [0, 1]]; s.mathMode = 'guess'; s.cam = wideCam(); },
        hint: 'Cell (i, j) = δᵢ · xⱼ.' },
    ],
  });

  /* ======================= drawing helpers ======================= */
  const inBox = (p, c, h) => Math.abs(p[0] - c[0]) <= h + 1e-9 && Math.abs(p[1] - c[1]) <= h + 1e-9;
  /** polyline restricted to an axis-aligned box (drops the parts outside) */
  function clipped(g, pts, c, h, opts) {
    let run = [];
    for (const p of pts) { if (inBox(p, c, h)) run.push(p); else { if (run.length > 1) g.polyline(run, opts); run = []; } }
    if (run.length > 1) g.polyline(run, opts);
  }
  function drawPanel(g, c, h, title, opts) {
    opts = opts || {};
    g.polygon([[c[0] - h, c[1] - h], [c[0] + h, c[1] - h], [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]], { fill: 'hud', alpha: opts.dim ? 0.02 : 0.05, stroke: 'grid', width: 1 });
    if (opts.grid) for (const u of [-1, -0.5, 0.5, 1]) { const k = u * (opts.unit || 1); g.line([c[0] + k, c[1] - h], [c[0] + k, c[1] + h], { color: 'grid', alpha: 0.45 }); g.line([c[0] - h, c[1] + k], [c[0] + h, c[1] + k], { color: 'grid', alpha: 0.45 }); }
    g.line([c[0] - h, c[1]], [c[0] + h, c[1]], { color: 'gridAxis', width: 1, alpha: opts.dim ? 0.25 : 0.6 });
    g.line([c[0], c[1] - h], [c[0], c[1] + h], { color: 'gridAxis', width: 1, alpha: opts.dim ? 0.25 : 0.6 });
    if (title) g.label(title, [c[0], c[1] + h], { anchor: 'above', dy: -4, size: opts.size || 12.5, color: opts.dim ? 'muted' : 'text' });
  }
  /** colored cells (2D heat via polygons) with printed numbers; M rows top → bottom */
  function cells(g, M, x0, y1, cw, ch, range, opts) {
    opts = opts || {};
    const r = M.length, cN = M[0].length;
    for (let i = 0; i < r; i++) for (let j = 0; j < cN; j++) {
      const a = opts.alphaAt ? opts.alphaAt(i, j) : 1; if (a <= 0) continue;
      const xa = x0 + j * cw, yb = y1 - (i + 1) * ch;
      g.polygon([[xa, yb], [xa + cw, yb], [xa + cw, yb + ch], [xa, yb + ch]], { fill: divColor(M[i][j] / range), alpha: 0.95 * a, stroke: opts.stroke || 'grid', width: 1 });
      if (opts.numbers !== false) g.label(LA.fmt(M[i][j]), [xa + cw / 2, yb + ch / 2], { size: opts.size || 11.5, color: 'text', bg: false, font: 'ui', alpha: a });
    }
  }
  /** bracketed matrix of numbers in world space, colored by column */
  function matTile(g, M, c, title, opts) {
    opts = opts || {};
    const r = M.length, cN = M[0].length, cw = opts.cw || 0.58, ch = 0.36, a = opts.alpha === undefined ? 1 : opts.alpha;
    const w = cN * cw, h = r * ch, x0 = c[0] - w / 2, y1 = c[1] + h / 2;
    for (let i = 0; i < r; i++) for (let j = 0; j < cN; j++) g.label(LA.fmt(M[i][j]), [x0 + (j + 0.5) * cw, y1 - (i + 0.5) * ch], { size: 12, color: opts.colors ? opts.colors(i, j) : 'text', bg: false, font: 'ui', alpha: a });
    const br = (x, s) => { g.line([x, y1], [x, y1 - h], { color: 'hud', width: 1.3, alpha: a }); g.line([x, y1], [x + s * 0.08, y1], { color: 'hud', width: 1.3, alpha: a }); g.line([x, y1 - h], [x + s * 0.08, y1 - h], { color: 'hud', width: 1.3, alpha: a }); };
    br(x0 - 0.04, 1); br(x0 + w + 0.04, -1);
    if (title) g.label(title, [c[0], y1], { anchor: 'above', dy: -6, size: 12.5, color: opts.titleColor || 'gold', alpha: a });
  }
  const colRole = (i, j) => j === 0 ? 'i' : 'j';
  /** tiny square at o with a green bottom side and a red left side */
  /** side of the tiny square so that its image under J stays readable inside a panel (shrinks when J is large) */
  const sideFor = J => E * Math.min(1, 0.85 / Math.max(1e-9, N.norm(N.col(J, 0)), N.norm(N.col(J, 1))));
  function square(g, o, a, dashed, e) {
    const E = e || E0;
    const q = [o, [o[0] + E, o[1]], [o[0] + E, o[1] + E], [o[0], o[1] + E]];
    g.line(q[0], q[1], { color: 'i', width: 2.2, alpha: a, dashed });
    g.line(q[3], q[0], { color: 'j', width: 2.2, alpha: a, dashed });
    g.line(q[1], q[2], { color: 'hud', width: 1, alpha: 0.7 * a, dashed: true });
    g.line(q[2], q[3], { color: 'hud', width: 1, alpha: 0.7 * a, dashed: true });
  }
  /** parallelogram of J (side E) at o with labeled column arrows */
  function para(g, o, J, a, name, e) {
    const E = e || E0;
    const u = N.scale(N.col(J, 0), E), v = N.scale(N.col(J, 1), E);
    g.parallelogram(u, v, { origin: o, fill: 'v', alpha: 0.2 * a, stroke: 'gold', width: 1.4, orient: true });
    g.vector(o, N.add(o, u), { color: 'i', width: 2.2, head: 8, alpha: a, label: name ? name + 'î' : undefined, labelOffset: 12, size: 11 });
    g.vector(o, N.add(o, v), { color: 'j', width: 2.2, head: 8, alpha: a, label: name ? name + 'ĵ' : undefined, labelOffset: -12, size: 11 });
  }
  function gArrow(g, o, v, label, a, opts) {
    opts = opts || {};
    let w = N.scale(v, GS); const nw = N.norm(w);
    if (nw > 1.1) { w = N.scale(w, 1.1 / nw); if (label) label += ' (clipped)'; }
    const tip = N.add(o, w);
    if (nw < 0.012) { g.point(o, { r: 3, color: 'v', alpha: a }); if (label) g.label(label + ' ≈ 0', o, { color: 'v', size: 11.5, dy: 14, alpha: a }); return; }
    g.vector(o, tip, { color: 'v', width: opts.width || 2.8, head: 10, alpha: a, label, labelOffset: opts.off === undefined ? 14 : opts.off, labelAt: opts.at });
  }

  /* ======================= the three-space network scene ======================= */
  function renderNet(g, s, ctx) {
    const f = fw(s), act = s.act;
    const titles = ['input x', 'h₁ = ' + act + '(W₁x + b₁)', 'h₂ = ' + act + '(W₂h₁ + b₂)'];
    for (let k = 0; k < 3; k++) drawPanel(g, [PX[k], PY[k]], PR, titles[k], { grid: k === 0 });
    // warped grids: the input grid pushed through layer 1 (panel 2) and through both layers (panel 3)
    const L1 = p => N.add(N.matVec(f.W1, p), B1).map(z => actF(act, z));
    const L2 = p => N.add(N.matVec(f.W2, p), B2).map(z => actF(act, z));
    for (const u of [-1, -0.5, 0, 0.5, 1]) {
      const a = [], b = [];
      for (let k = 0; k <= 16; k++) { const v = -PR + 2 * PR * k / 16; a.push([u, v]); b.push([v, u]); }
      for (const line of [a, b]) {
        const st = { color: u === 0 ? 'gridAxis' : 'grid', width: 1, alpha: 0.7 };
        const m1 = line.map(L1);
        clipped(g, m1.map(q => P(1, q)), [PX[1], PY[1]], PR, st);
        clipped(g, m1.map(L2).map(q => P(2, q)), [PX[2], PY[2]], PR, st);
      }
    }
    const X = P(0, s.x), H1 = P(1, clampL(f.h1, PR)), H2 = P(2, clampL(f.h2, PR));
    // forward travel
    const fv = s.fwd;
    if (fv >= 1) g.line(X, H1, { color: 'v', width: 1, dashed: [3, 4], alpha: 0.35 });
    if (fv >= 2) g.line(H1, H2, { color: 'v', width: 1, dashed: [3, 4], alpha: 0.35 });
    if (fv > 0 && fv < 1) g.point(lerpV(X, H1, fv), { r: 5, color: 'v', alpha: 0.8 });
    if (fv > 1 && fv < 2) g.point(lerpV(H1, H2, fv - 1), { r: 5, color: 'v', alpha: 0.8 });
    // output line under the h₂ panel
    if (fv > 2 || s.back > 0) {
      const grow = Math.min(1, Math.max(0, fv - 2));
      if (fv >= 2) g.line(H2, [OX, OY], { color: 'v', width: 1, dashed: [3, 4], alpha: 0.3 * grow });
      g.line([OX - 1.15, OY], [OX + 1.15, OY], { color: 'gridAxis', width: 1.2 });
      g.line([OX, OY - 0.1], [OX, OY + 0.1], { color: 'gridAxis', width: 1.2 });
      g.label('t = 0', [OX, OY - 0.1], { anchor: 'below', dy: 3, dx: -16, size: 11, color: 'muted' });
      g.label('output y', [OX - 1.15, OY], { anchor: 'right', dx: -6, size: 11.5, color: 'muted' });
      const yv = LA.clamp(f.y, -2.2, 2.2) * OS * grow;
      if (Math.abs(yv) > 0.005) g.line([OX, OY], [OX + yv, OY], { color: 'v', width: 7, alpha: 0.85 });
      if (grow > 0.5) { g.label('y = ' + LA.fmt(f.y), [OX + yv, OY], { anchor: 'below', dy: 9, dx: 18, color: 'v', size: 12 }); g.label('L = ' + LA.fmt(f.L), [OX - 0.75, OY - 0.45], { size: 12.5, color: 'gold' }); }
    }
    // local Jacobians (step 2, saturation, playground)
    if (s.sqA > 0) {
      const e1 = sideFor(f.J1), e2 = sideFor(f.J2);
      square(g, X, s.sqA, false, e1); para(g, H1, f.J1, s.sqA, 'J₁', e1);
      square(g, H1, 0.5 * s.sqA, true, e2); para(g, H2, f.J2, s.sqA, 'J₂', e2);
    }
    if (s.dBars) {
      [[f.D1, -2.45, 'D₁'], [f.D2, -1.15, 'D₂']].forEach(([D, x0, name]) => {
        g.bars(D, { origin: [x0, -1.7], width: 0.32, gap: 0.14, height: 0.9, max: 1, colors: ['i', 'j'], values: true });
        g.label(name + ': tanh slopes', [x0 + 0.39, -1.7], { anchor: 'below', dy: 6, size: 11.5, color: 'muted' });
      });
    }
    // composition hop (step 3)
    if (s.hop >= 0) {
      const e1 = [E, 0], e2 = [0, E];
      const shape = (o, M) => { const u = N.matVec(M, e1), v = N.matVec(M, e2); return [o, N.add(o, u), N.add(N.add(o, u), v), N.add(o, v)]; };
      const st = [shape(X, N.identity(2)), shape(H1, f.J1), shape(H2, f.J21)];
      const k = Math.min(1, Math.floor(s.hop)), u = s.hop - k;
      const cur = s.hop >= 2 ? st[2] : st[k].map((p, i) => lerpV(p, st[k + 1][i], seg(u, 0, 1, 'inOut')));
      for (let q = 0; q < 3; q++) if (s.hop >= q + 0.999 || q === 0) g.polygon(st[q], { fill: 'v', alpha: 0.06, stroke: 'gold', width: 1, back: true });
      g.polygon(cur, { fill: 'v', alpha: 0.22, stroke: null });
      g.line(cur[0], cur[1], { color: 'i', width: 2.6 }); g.line(cur[3], cur[0], { color: 'j', width: 2.6 });
      g.line(cur[1], cur[2], { color: 'gold', width: 1.2 }); g.line(cur[2], cur[3], { color: 'gold', width: 1.2 });
      if (s.prodA > 0) {
        g.polyline(st[2], { color: 'w', width: 2.2, dashed: [6, 4], closed: true, alpha: s.prodA });
        g.label('J₂J₁ (dashed)', st[2][1], { color: 'w', size: 12, anchor: 'below', dy: 10, alpha: s.prodA });
        if (s.prodA > 0.6) g.marker.check(H2, { dx: -26, dy: 26, size: 20 });
      }
    }
    // the chain as tiles (step 4)
    if (s.tiles > 0) {
      const A = k => LA.clamp(s.tiles - k, 0, 1);
      matTile(g, f.J1, [-2.05, -0.75], '1 · J₁ = D₁W₁', { colors: colRole, alpha: A(0), cw: 0.52 });
      matTile(g, f.J2, [-0.8, -0.75], '2 · J₂ = D₂W₂', { colors: colRole, alpha: A(1), cw: 0.52 });
      matTile(g, [W3], [-2.05, -1.55], '3 · w₃ᵀ', { colors: colRole, alpha: A(2), cw: 0.52 });
      matTile(g, [[f.g3]], [-0.8, -1.55], '4 · y − t', { colors: () => 'v', alpha: A(3) });
      if (s.tiles >= 3.9) {
        g.label('∂L/∂x = (y − t) · w₃ᵀ · J₂ · J₁', [-1.4, -2.1], { size: 13, color: 'text', bold: true });
        g.label('picture: left → right · formula: right → left', [-1.4, -2.42], { size: 11, color: 'muted' });
      }
    }
    if (s.gxStrip) {
      cells(g, [f.gx], -2.5, 0.22, 0.55, 0.36, 1, {});
      g.label('∂L/∂x: a 1×2 row', [-1.33, 0.04], { anchor: 'left', size: 11.5, color: 'v' });
    }
    // backward pull-back (step 5, saturation, playground)
    if (s.back > 0) {
      const b = s.back;
      const a0 = seg(b, 0, 0.2);
      if (a0 > 0) {
        const len = LA.clamp(f.g3, -2.1, 2.1) * OS * a0;
        if (Math.abs(len) > 0.01) g.vector([OX, OY + 0.28], [OX + len, OY + 0.28], { color: 'v', width: 2.6, head: 9, label: '∂L/∂y', labelOffset: -12, labelAt: 0.5 });
      }
      const a1 = seg(b, 0.2, 0.45), m2 = seg(b, 0.45, 0.55);
      if (a1 > 0) {
        if (m2 > 0) gArrow(g, H2, f.gh2, undefined, 0.3, { width: 1.6 });
        gArrow(g, H2, N.scale(lerpV(f.gh2, f.d2, m2), a1), m2 < 0.5 ? '∂L/∂h₂' : 'δ₂', 1);
      }
      const a2 = seg(b, 0.55, 0.7), m1 = seg(b, 0.7, 0.8);
      if (a2 > 0) {
        g.line(H2, H1, { color: 'mag', width: 1.4, dashed: [5, 4], alpha: 0.7 * a2 });
        g.label('W₂ᵀ', lerpV(H2, H1, 0.5), { color: 'mag', size: 12, dy: -12, alpha: a2 });
        if (m1 > 0) gArrow(g, H1, f.gh1, undefined, 0.3, { width: 1.6 });
        gArrow(g, H1, N.scale(lerpV(f.gh1, f.d1, m1), a2), m1 < 0.5 ? '∂L/∂h₁' : 'δ₁', 1);
      }
      const a3 = seg(b, 0.8, 1);
      if (a3 > 0) {
        g.line(H1, X, { color: 'mag', width: 1.4, dashed: [5, 4], alpha: 0.7 * a3 });
        g.label('W₁ᵀ', lerpV(H1, X, 0.5), { color: 'mag', size: 12, dy: -12, alpha: a3 });
        gArrow(g, X, N.scale(f.gx, a3), '∂L/∂x', 1);
      }
    }
    // why the transpose (step 6)
    if (s.tdot > 0) {
      const a = s.tdot, d = [0.3, 0.1], Jd = N.matVec(f.J1, d);
      // steps d drawn ×2 for visibility (both sides of the identity use the same scale)
      g.vector(X, N.add(X, N.scale(d, 2)), { color: 'w', width: 2.8, head: 9, alpha: a, label: 'd', labelOffset: 16 });
      g.vector(H1, N.add(H1, N.scale(Jd, 2)), { color: 'w', width: 2.4, head: 9, alpha: a, dashed: true, label: 'J₁d', labelOffset: -18 });
      gArrow(g, H1, f.gh1, 'g', a, { off: 16 });
      gArrow(g, X, f.gx, 'J₁ᵀg', a, { off: -20, at: 0.55 });
      g.label('steps drawn ×2, gradients ×0.5', [PX[0], PY[0] - PR], { anchor: 'below', dy: 4, size: 10.5, color: 'muted', alpha: a });
      const bp = g.P.toScreen([-2.45, -0.35]);
      g.bars([N.dot(Jd, f.gh1), N.dot(d, f.gx)], { screen: true, x: bp[0], y: bp[1], w: 170, h: 90, gap: 24, max: 0.3, colors: ['w', 'v'], labels: ['(J₁d)·g', 'd·(J₁ᵀg)'], values: true, title: 'slope after = slope before', alpha: 0.9 * a });
    }
    // outer product (step 7) and the GD step (ML tie-in)
    if (s.outerA > 0) renderOuter(g, s, f);
    if (s.showOuter) {
      cells(g, f.dW1, -2.65, -0.3, 0.5, 0.45, 0.8, {}); g.label('∂L/∂W₁ = δ₁xᵀ', [-2.15, -0.3], { anchor: 'above', dy: -4, size: 11.5, color: 'gold' });
      cells(g, f.dW2, -1.25, -0.3, 0.5, 0.45, 0.8, {}); g.label('∂L/∂W₂ = δ₂h₁ᵀ', [-0.75, -0.3], { anchor: 'above', dy: -4, size: 11.5, color: 'gold' });
    }
    if (s.mlTie) renderGD(g, s, f);
    // saturation inset + readouts
    if (s.satInset) {
      const zs = [], lp = g.P.toScreen([-2.6, -0.2]);
      for (let k = 0; k <= 40; k++) { const z = -6 + 12 * k / 40; zs.push([z, Math.tanh(z)]); }
      g.lineChart([{ points: zs, color: 'teal' }, { points: [[LA.clamp(f.z1[0], -6, 6), -1.1], [LA.clamp(f.z1[0], -6, 6), 1.1]], color: 'i' }, { points: [[LA.clamp(f.z1[1], -6, 6), -1.1], [LA.clamp(f.z1[1], -6, 6), 1.1]], color: 'j', dash: [4, 3] }],
        { x: lp[0], y: lp[1], w: 220, h: 90, xRange: [-6, 6], yRange: [-1.15, 1.15], title: 'tanh(z) at z₁: unit 1 green, unit 2 red' });
      g.label('‖∂L/∂x‖ = ' + LA.fmt(N.norm(f.gx)), [-2.65, -1.75], { anchor: 'left', size: 12.5, color: 'v' });
      g.label('δ₁ = (' + (Math.abs(f.d1[0]) < 0.01 ? LA.fmt.fixed(f.d1[0], 5) : LA.fmt(f.d1[0])) + ', ' + (Math.abs(f.d1[1]) < 0.01 ? LA.fmt.fixed(f.d1[1], 5) : LA.fmt(f.d1[1])) + ')', [-2.65, -2.05], { anchor: 'left', size: 12.5, color: 'v' });
    }
    // probe + handle
    g.point(X, { r: 6, color: 'v', label: 'x', dy: -12, dx: -12 });
    if (fv >= 1) g.point(H1, { r: 5, color: 'v', label: 'h₁', dy: -12, dx: -12 });
    if (fv >= 2) g.point(H2, { r: 5, color: 'v', label: 'h₂', dy: -12, dx: -12 });
    g.handle('x', X, (p, st) => { st.x = [LA.clamp(p[0] - PX[0], -1.2, 1.2), LA.clamp(p[1] - PY[0], -1.2, 1.2)]; }, { color: 'v' });
  }

  /** step 7: δ₂ column × h₁ᵀ row → 2×2 heatmap filling cell by cell, then the rank-1 collapse inset */
  function renderOuter(g, s, f) {
    const a = s.outerA, x0 = -2.12, y1 = -0.28, cw = 0.5, ch = 0.5;
    cells(g, f.d2.map(v => [v]), -2.62, y1, 0.4, ch, 0.8, { size: 10.5 });
    g.label('δ₂', [-2.42, y1 - 2 * ch], { anchor: 'below', dy: 4, size: 14, color: 'v', bold: true });
    cells(g, [f.h1], x0, y1 + 0.4, cw, 0.34, 0.8, { size: 10.5 });
    g.label('h₁ᵀ', [x0 + 2 * cw + 0.06, y1 + 0.23], { anchor: 'left', size: 13, color: 'text', bold: true });
    cells(g, f.dW2, x0, y1, cw, ch, 0.8, { size: 10.5, alphaAt: (i, j) => LA.clamp(a * 4.2 - (2 * i + j), 0, 1) });
    g.polyline([[x0, y1], [x0 + 2 * cw, y1], [x0 + 2 * cw, y1 - 2 * ch], [x0, y1 - 2 * ch]], { color: 'gold', width: 1.4, closed: true });
    g.label('∂L/∂W₂ = δ₂h₁ᵀ', [x0 + cw, y1 - 2 * ch], { anchor: 'below', dy: 6, size: 12, color: 'gold' });
    // collapse inset: the plane under lerp(I, Â, u), Â = δ₂h₁ᵀ rescaled for visibility
    const c0 = [-0.55, -0.85], h = 0.42, u = s.collapse;
    drawPanel(g, c0, h, u > 0.5 ? 'rank 1' : 'on the plane', { size: 11 });
    if (u > 0) {
      const A = N.scaleM(f.dW2, 1.0 / (N.frobenius(f.dW2) || 1)), M = lerpM(N.identity(2), A, u);
      for (const k of [-2, -1, 0, 1, 2]) {
        const l1 = [], l2 = [];
        for (let q = 0; q <= 8; q++) { const v = -0.4 + 0.8 * q / 8; l1.push(N.add(c0, N.matVec(M, [k * 0.18, v]))); l2.push(N.add(c0, N.matVec(M, [v, k * 0.18]))); }
        clipped(g, l1, c0, h, { color: 'grid', width: 1.1 }); clipped(g, l2, c0, h, { color: 'grid', width: 1.1 });
      }
      const dn = N.normalize(f.d2);
      g.line(N.add(c0, N.scale(dn, -h * 0.95)), N.add(c0, N.scale(dn, h * 0.95)), { color: 'gold', width: 1.6, dashed: [5, 4], alpha: u });
    }
  }

  /** ML tie-in: W₂ heatmap shifting by −η·δ₂h₁ᵀ, a 12×12 rank-1 stand-in, the loss readout */
  function renderGD(g, s, f) {
    const B = batchData();
    const x0 = -2.4, y1 = -0.3;
    cells(g, s.W2, x0, y1, 0.6, 0.5, 1.2, {});
    g.polyline([[x0, y1], [x0 + 1.2, y1], [x0 + 1.2, y1 - 1], [x0, y1 - 1]], { color: 'gold', width: 1.4, closed: true });
    g.label('W₂ ← W₂ − η·δ₂h₁ᵀ', [x0 + 0.6, y1], { anchor: 'above', dy: -5, size: 12.5, color: 'gold' });
    g.label('L: ' + LA.fmt(s._L0 === undefined ? f.L : s._L0) + ' → ' + LA.fmt(f.L), [x0 + 0.6, -1.65], { size: 15, color: 'gold', bold: true });
    const R = Math.max(...B.thumb.map(r => Math.max(...r.map(Math.abs))));
    const bd = { x0: -0.95, x1: -0.15, y0: -1.15, y1: -0.35 };
    g.heat2d(B.thumb, bd, { range: [-R, R], ramp: divRamp, smooth: false, alpha: 0.95 });
    g.polyline([[bd.x0, bd.y0], [bd.x1, bd.y0], [bd.x1, bd.y1], [bd.x0, bd.y1]], { color: 'grid', closed: true, width: 1 });
    g.label('rank-1 update', [-0.55, bd.y1], { anchor: 'above', dy: -4, size: 11, color: 'text' });
    g.label('12×12 stand-in for a 4096×4096 W (synthetic)', [-2.7, -2.1], { anchor: 'left', size: 11, color: 'muted' });
    g.label('see p6-08', [-2.7, -2.4], { anchor: 'left', size: 11.5, color: 'muted' });
  }

  /* ======================= batch of rank-1 sheets ======================= */
  function renderBatch(g, s) {
    const B = batchData(), u = s.batchU, n = Math.min(4, Math.floor(u * 4 + 1e-6));
    const tw = 1.8, th = 1.35, xs = [-3.1, -1.03, 1.03, 3.1], ty0 = 2.45;
    const sw = 3.6, sh = 2.7, sx0 = -1.8, sy0 = -2.25;
    const frame = (x0, y0, w, h, color) => g.polyline([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], { color: color || 'grid', closed: true, width: 1 });
    for (let k = 0; k < 4; k++) {
      const x0 = xs[k] - tw / 2;
      g.heat2d(B.sheets[k], { x0, x1: x0 + tw, y0: ty0, y1: ty0 + th }, { range: [-B.r1, B.r1], ramp: divRamp, smooth: false, alpha: k < n ? 0.7 : 0.95 });
      frame(x0, ty0, tw, th);
      g.label('δ⁽' + sup(k + 1) + '⁾x⁽' + sup(k + 1) + '⁾ᵀ · rank 1', [xs[k], ty0 + th], { anchor: 'above', dy: -4, size: 11.5, color: 'text' });
      const mk = LA.clamp(u * 4 - k, 0, 1);
      if (mk > 0 && mk < 1) {
        const e = seg(mk, 0, 1, 'inOut'), x = LA.lerp(x0, sx0, e), y = LA.lerp(ty0, sy0, e), w = LA.lerp(tw, sw, e), h = LA.lerp(th, sh, e);
        g.heat2d(B.sheets[k], { x0: x, x1: x + w, y0: y, y1: y + h }, { range: [-B.r1, B.r1], ramp: divRamp, smooth: false, alpha: 0.5 });
      }
    }
    g.label('one sample = stripes: every row is a multiple of x⁽¹⁾ᵀ', [xs[0] - tw / 2, ty0 - 0.1], { anchor: 'left', dy: 12, size: 11, color: 'muted' });
    if (n > 0) g.heat2d(B.sums[n - 1], { x0: sx0, x1: sx0 + sw, y0: sy0, y1: sy0 + sh }, { range: [-B.r2, B.r2], ramp: divRamp, smooth: false, alpha: 0.95 });
    frame(sx0, sy0, sw, sh, 'gold');
    g.label(n ? 'batch gradient = sum of ' + n + ' sheet' + (n > 1 ? 's' : '') + ' · rank ' + B.ranks[n - 1] + ' (≤ 4)' : 'batch gradient (6×8): empty', [0, sy0 + sh], { anchor: 'above', dy: -5, size: 13, color: 'gold', bold: true });
    g.label('synthetic: δ ∈ ℝ⁶, x ∈ ℝ⁸ per sample (seed 506)', [0, sy0], { anchor: 'below', dy: 6, size: 11, color: 'muted' });
    g.hud('The picture stops at 3. The formula does not.', 18, g.size.h - 24, { size: 14, color: 'text' });
  }

  /* ======================= deep stack ======================= */
  function renderDeep(g, s, ctx) {
    const r = deepRun(s), d = s.depth;
    const cen = k => [DX[k % 3], DY[Math.floor(k / 3)]];
    for (let k = 0; k <= 8; k++) {
      const c = cen(k), on = k <= d;
      drawPanel(g, c, DH, 'h' + sub(k) + (k === 0 ? ' (input)' : k === d ? ' (output)' : ''), { dim: !on, size: 12 });
      if (!on) continue;
      if (k <= s.deepF + 1e-6) {
        const pts = r.circles[k].map(p => N.add(c, N.scale(p, DU))); pts.push(pts[0]);
        clipped(g, pts, c, DH, { color: k === 0 ? 'v' : 'teal', width: 1.8 });
        const rad = Math.max(...r.circles[k].map(p => N.norm(p)));
        g.label('r = ' + (rad < 0.1 ? LA.fmt.fixed(rad, 3) : LA.fmt(rad)) + (rad * DU > DH ? ' (off the panel)' : ''), [c[0] - DH + 0.06, c[1] - DH + 0.06], { anchor: 'left', dy: -9, size: 10.5, color: 'teal', bg: false });
        const pk = r.pts[k]; if (inBox(N.add(c, N.scale(pk, DU)), c, DH)) g.point(N.add(c, N.scale(pk, DU)), { r: 3.5, color: 'v' });
      }
      if (k >= d - s.deepB - 1e-6) {
        const gk = r.gs[k], pk = N.add(c, N.scale(clampL(r.pts[k], 1.3), DU)), len = N.norm(gk) * 0.9;
        const v = len > 1.0 ? N.scale(gk, 1.0 / N.norm(gk)) : N.scale(gk, 0.9);
        if (len > 0.012) g.vector(pk, N.add(pk, v), { color: 'v', width: 2.4, head: 9, label: 'g' + sub(k), labelOffset: 12 });
        else g.point(pk, { r: 3, color: 'v' });
        g.label('‖g‖ = ' + (r.norms[k] < 0.1 ? LA.fmt.fixed(r.norms[k], 3) : LA.fmt(r.norms[k])) + (len > 1.0 ? ' (clipped)' : ''), [c[0] + DH - 0.06, c[1] - DH + 0.06], { anchor: 'right', dy: -9, size: 10.5, color: 'v', bg: false });
      }
    }
    // weight-scale track with its knob (the handle used in check 1)
    const kx = -3 + (LA.clamp(s.c, 0.2, 2) - 0.2) / 1.8 * 6;
    g.line([-3, TY], [3, TY], { color: 'gridAxis', width: 2 });
    const x1 = -3 + 0.8 / 1.8 * 6; g.line([x1, TY - 0.1], [x1, TY + 0.1], { color: 'gridAxis', width: 1.4 }); g.label('c = 1', [x1, TY - 0.1], { anchor: 'below', dy: 4, size: 10.5, color: 'muted' });
    g.point([kx, TY], { r: 6, color: 'gold' });
    g.label('weight scale c = ' + LA.fmt(s.c) + ' · W = c·R(30°) · ' + s.act, [3, TY + 0.1], { anchor: 'right', dy: -10, size: 11.5, color: 'gold' });
    g.handle('c', [kx, TY], (p, st) => { st.c = Math.round(LA.clamp(0.2 + (p[0] + 3) / 6 * 1.8, 0.2, 2) * 100) / 100; }, { color: 'gold' });
    // norm bars
    const vals = [], cols = [], labs = [];
    for (let k = 0; k <= d; k++) { const shown = k >= d - s.deepB - 1e-6; const v = shown ? r.norms[k] : 0; vals.push(Math.min(v, 2)); cols.push(v > 2 ? 'bad' : 'gold'); labs.push('g' + sub(k)); }
    const Wd = g.size.w, Hh = g.size.h;
    g.bars(vals, { screen: true, x: 40, y: Hh - 70, w: Wd - 80, h: 38, max: 2, colors: cols, labels: labs, title: '‖gₖ‖ at every layer (bars clipped at 2; output gradient has length 1)' });
    if (r.norms[0] > 2 && s.deepB >= d - 1e-6) g.hud('↑ off the chart: ‖g₀‖ = ' + LA.fmt(r.norms[0]), Wd - 40, Hh - 82, { anchor: 'right', size: 11.5, color: 'bad' });
    if (s.cross) { g.label('“sizes stay about the same”', [-3.9, -2.62], { color: 'muted', size: 14, anchor: 'left', dx: 24 }); g.marker.cross([-3.9, -2.62], { size: 20, dx: 8 }); }
  }

  /* ======================= check 3 scene: δxᵀ typed by the learner ======================= */
  function renderOuterCheck(g, s) {
    const M = s.guessOuter, x0 = -2.6, y1 = 2.6, cw = 0.75;
    cells(g, [[2], [-1]], x0 - 0.62, y1, 0.45, cw, 6, {}); g.label('δ', [x0 - 0.7, y1 - cw], { anchor: 'right', dx: -4, size: 15, color: 'v', bold: true });
    cells(g, [[3, 1]], x0, y1 + 0.55, cw, 0.42, 6, {}); g.label('xᵀ', [x0 + cw, y1 + 0.55], { anchor: 'above', dy: -4, size: 14, color: 'text', bold: true });
    cells(g, M, x0, y1, cw, cw, 6, {});
    g.polyline([[x0, y1], [x0 + 2 * cw, y1], [x0 + 2 * cw, y1 - 2 * cw], [x0, y1 - 2 * cw]], { color: 'w', width: 1.6, closed: true });
    g.label('your δxᵀ: cell (i, j) = δᵢ·xⱼ', [x0 + cw, y1 - 2 * cw], { anchor: 'below', dy: 6, size: 12, color: 'w' });
    // the plane under your matrix (scaled to fit) and the true rank-1 image line (faint gold)
    const c0 = [1.75, 0.4], h = 1.9;
    drawPanel(g, c0, h, 'your matrix acting on the plane (scaled to fit)', { size: 11.5 });
    const fr = N.frobenius(M) || 1, A = N.scaleM(M, 1.6 / Math.max(1, fr));
    for (const k of [-3, -2, -1, 0, 1, 2, 3]) {
      const l1 = [], l2 = [];
      for (let q = 0; q <= 10; q++) { const v = -1.6 + 3.2 * q / 10; l1.push(N.add(c0, N.matVec(A, [k * 0.5, v]))); l2.push(N.add(c0, N.matVec(A, [v, k * 0.5]))); }
      clipped(g, l1, c0, h, { color: k === 0 ? 'gridAxis' : 'grid', width: 1.1 }); clipped(g, l2, c0, h, { color: k === 0 ? 'gridAxis' : 'grid', width: 1.1 });
    }
    const dn = N.normalize([2, -1]);
    g.line(N.add(c0, N.scale(dn, -h)), N.add(c0, N.scale(dn, h)), { color: 'gold', width: 1.4, dashed: [5, 4], alpha: 0.45 });
    g.label('true answer collapses onto the δ line', N.add(c0, [0, -h]), { anchor: 'below', dy: 6, size: 11, color: 'gold' });
    const det = N.det(M);
    if (Math.abs(det) < 0.02 * fr * fr && fr > 0.1) g.label('rank 1: the plane collapsed to a line', [c0[0], c0[1] + h - 0.3], { size: 12, color: 'ok' });
  }

  /* ======================= recap: three mini-cards ======================= */
  function renderRecap(g, s) {
    const f = fw(s);
    const A = [[1.1, 0.3], [0.2, 0.9]], Bm = [[0.8, -0.45], [0.5, 1.0]], BA = N.matMul(Bm, A), e = 0.42;
    const shp = (o, M) => { const u = N.matVec(M, [e, 0]), v = N.matVec(M, [0, e]); return [o, N.add(o, u), N.add(N.add(o, u), v), N.add(o, v)]; };
    const cards = [
      { title: 'chain = one product', cam: { center: [0.9, 0.35], halfH: 1.5 }, draw(g2) {
        const s0 = shp([-0.05, 0.1], N.identity(2)), s1 = shp([0.6, 0.1], A), s2 = shp([1.45, 0.1], BA);
        [s0, s1, s2].forEach((q, i) => { g2.polygon(q, { fill: 'v', alpha: 0.2, stroke: null }); g2.line(q[0], q[1], { color: 'i', width: 2.2 }); g2.line(q[3], q[0], { color: 'j', width: 2.2 }); g2.line(q[1], q[2], { color: 'gold', width: 1 }); g2.line(q[2], q[3], { color: 'gold', width: 1 }); });
        g2.polyline(s2, { color: 'w', width: 2, dashed: [5, 4], closed: true });
        g2.label('J₁', [0.47, 0.85], { size: 12, color: 'hud' }); g2.label('J₂', [1.25, 0.95], { size: 12, color: 'hud' }); g2.label('= J₂J₁', [1.6, -0.25], { size: 12, color: 'w' });
      } },
      { title: 'pull back by Jᵀ', cam: { center: [0.9, 0.35], halfH: 1.5 }, draw(g2) {
        const ps = [[1.65, -0.2], [0.9, 0.2], [0.15, -0.2]], gs = [[-0.05, 0.95], [0.3, 0.55], [0.12, 0.25]], ls = ['g₂', 'g₁', 'g₀'], ns = ['h₂', 'h₁', 'x'];
        g2.line(ps[0], ps[1], { color: 'mag', dashed: [4, 4], width: 1.2 }); g2.line(ps[1], ps[2], { color: 'mag', dashed: [4, 4], width: 1.2 });
        ps.forEach((p, i) => { g2.point(p, { r: 3.5, color: 'v' }); g2.label(ns[i], p, { anchor: 'below', dy: 6, size: 11, color: 'muted' }); g2.vector(p, N.add(p, gs[i]), { color: 'v', width: 2.4, head: 8, label: ls[i], labelOffset: 12, size: 11 }); });
        g2.label('W₂ᵀ', [1.3, -0.25], { size: 11, color: 'mag' }); g2.label('W₁ᵀ', [0.5, -0.25], { size: 11, color: 'mag' });
      } },
      { title: 'δxᵀ: rank 1', cam: { center: [0.55, 0.55], halfH: 1.5 }, draw(g2) {
        cells(g2, f.d2.map(v => [v]), -0.3, 1.0, 0.32, 0.55, 0.8, { size: 10 });
        cells(g2, [f.h1], 0.12, 1.45, 0.55, 0.32, 0.8, { size: 10 });
        cells(g2, f.dW2, 0.12, 1.0, 0.55, 0.55, 0.8, { size: 10 });
        g2.label('δ₂', [-0.15, -0.2], { size: 12, color: 'v' }); g2.label('h₁ᵀ', [0.95, 1.62], { size: 12, color: 'text', anchor: 'left' });
        g2.label('∂L/∂W₂', [0.67, -0.25], { size: 12, color: 'gold' });
      } },
    ];
    const Wd = g.size.w, Hh = g.size.h, pad = 10, cw = (Wd - pad * 4) / 3, ch = Math.min(Hh * 0.6, cw * 1.25), y = (Hh - ch) / 2;
    cards.forEach((cd, i) => {
      const r = { x: pad + i * (cw + pad), y, w: cw, h: ch };
      g.viewport(r, { view: '2d', cam: Object.assign(LA.cam.defaults2d(), cd.cam), bg: 'panel' }, (g2, rp) => { cd.draw(g2); g2.label(cd.title, [rp.x + rp.w / 2, rp.y + 16], { screen: true, size: 12.5, color: 'text', bold: true }); });
    });
  }
})();
