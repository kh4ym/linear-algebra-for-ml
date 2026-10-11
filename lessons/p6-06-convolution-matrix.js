/* p6-06 — Convolution is a matrix multiplication: a sliding 3-tap filter is a banded 6×8 matrix; stride drops rows,
   padding adds columns, the transpose runs it backward, and a 3×3 image kernel is a 9×25 band pattern. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, fmt = LA.fmt, lerp = LA.lerp;
  const X0 = [0, 0, 1, 1, 1, 0, 0, 2];                    // an edge up, an edge down, a spike
  const EDGE = [1, 0, -1], BLUR = [0.25, 0.5, 0.25];
  const SOBEL = [[1, 0, -1], [2, 0, -2], [1, 0, -1]];
  const IMG = [0, 1, 2, 3, 4].map(i => [0, 1, 2, 3, 4].map(j => (i >= 1 && i <= 3 && j >= 1 && j <= 3 ? 1 : 0)));
  const SUBS = ['₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉', '₁₀'];
  const sub = i => SUBS[i] || String(i + 1);
  // seeded perturbations for "untie the weights" (8 rows × 10 padded columns)
  const PERT = (function () { const r = LA.rng(607), M = []; for (let i = 0; i < 8; i++) { const row = []; for (let j = 0; j < 10; j++) row.push(+(LA.rng.gauss(r) * 0.55).toFixed(2)); M.push(row); } return M; })();
  // multi-channel banded sketch for the ML step (3 output × 4 input channels, 8×8 'same' blocks)
  const BIG = (function () { const r = LA.rng(608), M = []; const taps = []; for (let a = 0; a < 3; a++) { taps.push([]); for (let b = 0; b < 4; b++) taps[a].push([LA.rng.gauss(r), LA.rng.gauss(r), LA.rng.gauss(r)]); } for (let i = 0; i < 24; i++) { const row = []; for (let j = 0; j < 32; j++) { const a = Math.floor(i / 8), b = Math.floor(j / 8), ii = i % 8, jj = j % 8, k = jj - ii + 1; row.push(k >= 0 && k <= 2 ? taps[a][b][k] : 0); } M.push(row); } return M; })();

  /** C in padded column coordinates: R × L, with P = 1 ghost column at each end when padding is on. */
  function convC(f, stride, pad, untie) {
    const n = 8, P = pad ? 1 : 0, L = n + 2 * P, R = Math.floor((L - 3) / stride) + 1, C = [];
    for (let r = 0; r < R; r++) {
      const row = new Array(L).fill(0), p = r * stride;
      for (let k = 0; k < 3; k++) row[p + k] = f[k];
      if (untie) for (let c = 0; c < L; c++) row[c] += untie * PERT[r][c];
      C.push(row);
    }
    return { C, R, L, P };
  }
  const realC = cv => cv.C.map(row => row.slice(cv.P, cv.P + 8));
  const outY = (cv, x) => realC(cv).map(row => N.dot(row, x));
  const stride = s => (s.stride2 ? 2 : 1);
  const params = s => (s.untie > 0.5 ? convC(s.f, stride(s), s.pad).R * 8 : 3);
  // 2D: 9×25 matrix of the Sobel kernel on the flattened 5×5 image
  const C2D = (function () { const M = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { const row = new Array(25).fill(0); for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) row[(i + a) * 5 + (j + b)] = SOBEL[a][b]; M.push(row); } return M; })();
  const FLAT = [].concat.apply([], IMG);
  const Y2D = C2D.map(row => N.dot(row, FLAT));            // [−3,0,3,−4,0,4,−3,0,3]

  // ---- layout (world units; one cell = 1) ----
  const XC = j => j - 3.5;                                 // x-center of column j (j = −1 … 8 with padding ghosts)
  const YX = 4.6, BAR = 0.7;                               // baseline of the x bars, height per unit
  const YC = 3.3;                                          // top edge of C
  const YCOL = 5.9;                                        // x-center of the y column
  const div = v => LA.color.diverge(LA.clamp(v / 2, -1, 1));
  const camMain = () => ({ center: [1, 0.3], halfH: 6.6, halfW: 6.4 });
  const camImg = () => ({ center: [0.1, 0.9], halfH: 6.2, halfW: 6.7 });

  LA.registerLesson({
    id: 'p6-06-convolution-matrix',
    part: 6,
    title: 'Convolution is a matrix multiplication',
    subtitle: 'A sliding filter is a banded matrix with equal diagonals.',
    minutes: 10,
    prereqs: ['p2-02-matrix-vector', 'p6-01-batches-shapes-tensors', 'p5-06-chain-rule-backprop', 'p2-10-transpose-outer-symmetric'],
    tags: ['extension', 'capstone'],
    badge: { drawnIn: '2D (signals, matrices, a 5×5 image)', realSize: 'a 3×3 conv 64→128 (ResNet-style) has 3·3·64·128 = 73,728 shared weights; on a 224×224 input (stride 1, same padding) it acts as a ~6·10⁶ × 3·10⁶ matrix that is never formed', transfers: 'banded + shared structure, y = Cx, Cᵀ for the backward pass, parameter count independent of input size — exact at any size' },
    cantShow: ['Multi-channel stacks beyond three slices.', 'The full 10⁶×10⁶ matrix of a real layer (only its structure).', 'The Fourier view of convolution.'],
    terms: [
      { term: 'filter', def: 'The short list of weights (taps) slid along the input.' },
      { term: 'cross-correlation', def: 'The convention ML frameworks call “convolution”: output i = Σₖ fₖ x_{i+k}, no flipping.' },
      { term: 'banded matrix', def: 'A matrix whose nonzero entries sit on a few diagonals; when each diagonal is constant it is called Toeplitz.' },
      { term: 'weight sharing', def: 'The same tap value repeated along a whole diagonal of the matrix.' },
      { term: 'stride', def: 'How many positions the filter jumps between outputs; stride 2 drops every other row of C.' },
      { term: 'padding', def: 'Zeros added at the ends of the input so the output keeps the input’s length.' },
      { term: 'transposed convolution', def: 'Multiplication by Cᵀ, which spreads each output back over the inputs it came from.' },
    ],
    atlas: ['cnn', 'backprop'],
    thumbnail: 1,

    state() {
      return {
        view: '2d', cam: camMain(), scene: 'sig', panel: 'slide',
        x: X0.slice(), f: EDGE.slice(), pos: 0, stride2: false, pad: false, untie: 0, twoD: false,
        showC: false, fillRows: 0, fillY: 0, showTiles: false, rowHi: -1, band: -1, flip: 0, pull: 0,
        compact: 0, padFade: 0, pick: [-2.5, 2.8], toggles: false, cross: false,
        unroll: 0, fill2d: 0, reshape: 0,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'ml') return renderML(g, s);
      if (s.scene === 'recap') return renderRecap(g, s);
      // full-canvas viewport whose camera is widened so the whole layout fits any canvas aspect
      const W = g.size.w, H = g.size.h, cam = Object.assign({}, s.twoD && s.scene !== 'img' ? camImg() : s.cam);
      const hw = s.scene === 'img' || s.twoD ? 6.7 : 6.4;
      // canvas switches (top right, y up to ≈ 7.5) and the 8-row padded matrix + counter (down to ≈ −6.5) must fit
      if (s.toggles && s.scene === 'sig' && !s.twoD) { cam.center = [cam.center[0], 0.45]; cam.halfH = Math.max(cam.halfH, 7.35); }
      cam.halfH = Math.max(cam.halfH, hw * H / W);
      g.viewport({ x: 0, y: 0, w: 1, h: 1 }, { view: '2d', cam, border: false }, (g2) => {
        if (s.scene === 'img') renderImg(g2, s);
        else if (s.twoD) renderImg(g2, Object.assign({}, s, { unroll: 1, fill2d: 1, reshape: 1 }));
        else renderSig(g2, s, ctx);
      });
    },

    math(s, tex) {
      const H = str => tex.text(str, { html: true });
      const cv = convC(s.f, stride(s), s.pad, s.untie);
      const y = outY(cv, s.x);
      const yT = tex.text('[' + y.map(v => fmt(v)).join(', ') + ']', { color: 'b' });
      switch (s.panel) {
        case 'slide': return tex.lines(H('y<sub>i</sub> = Σ<sub>k</sub> f<sub>k</sub> x<sub>i+k</sub> = x<sub>i</sub> − x<sub>i+2</sub>'), tex.eq(tex.sym('y', { color: 'b' }), tex.op('='), yT));
        case 'matrix': return tex.block(tex.eq(tex.sym('y', { color: 'b' }), tex.op('='), tex.sym('C'), tex.sym('x', { color: 'v' })), H('C ∈ ℝ<sup>' + cv.R + '×8</sup>'), tex.eq(tex.mat(realC(cv), { colors: (i, j) => (realC(cv)[i][j] > 0 ? 'teal' : realC(cv)[i][j] < 0 ? 'mag' : 'muted') })));
        case 'row': return tex.lines(H('y<sub>4</sub> = C<sub>4,·</sub> · x = [0, 0, 0, 1, 0, −1, 0, 0] · x'), tex.eq(H('= x<sub>4</sub> − x<sub>6</sub> = 1 − 0 ='), tex.text('1', { color: 'b' })));
        case 'shared': return tex.lines(H('C<sub>i, i+k</sub> = f<sub>k</sub>'), tex.text('3 parameters · 18 band cells · 48 cells'));
        case 'blur': return tex.lines(tex.eq(tex.sym('f'), tex.op('='), tex.text('[' + s.f.map(v => fmt(v)).join(', ') + ']', { color: 'gold' })), tex.eq(tex.sym('y', { color: 'b' }), tex.op('='), yT));
        case 'untie': return tex.lines(tex.text('tied: 3 parameters'), tex.eq(tex.text('untied: 6·8 ='), tex.text('48', { color: 'bad' }), tex.text('parameters')), tex.eq(tex.text('now:'), tex.text(String(params(s)), { color: 'gold' })));
        case 'transpose': return tex.lines(H('∂L/∂x = C<sup>ᵀ</sup> ∂L/∂y'), H('C<sup>ᵀ</sup> ∈ ℝ<sup>8×6</sup>; e<sub>3</sub> → (1, 0, −1) on x<sub>3</sub>, x<sub>4</sub>, x<sub>5</sub>'));
        case 'stride': return tex.lines(H('stride 2: C ∈ ℝ<sup>3×8</sup>'), H('‘same’ padding: C ∈ ℝ<sup>8×8</sup>'), tex.eq(tex.text('now'), H('C ∈ ℝ<sup>' + (s.compact > 0.95 ? Math.ceil(cv.R / 2) : cv.R) + '×8</sup>')));
        case 'img': return tex.lines(H('y = C<sub>2d</sub> x, C<sub>2d</sub> ∈ ℝ<sup>9×25</sup>'), H('reshape (9) → (3, 3): [[−3, 0, 3], [−4, 0, 4], [−3, 0, 3]]'));
        case 'ml': return tex.lines(H('∂L/∂f = Σ<sub>positions</sub> δ<sub>i</sub> · window(x)<sub>i</sub>'), tex.text('73,728 weights · ~6·10⁶ × 3·10⁶ matrix'));
        case 'play': return tex.block(tex.eq(tex.sym('y', { color: 'b' }), tex.op('='), tex.sym('C'), tex.sym('x', { color: 'v' }), tex.op('='), yT), H('C ∈ ℝ<sup>' + cv.R + '×8</sup>'), tex.eq(tex.text('parameters'), tex.text(String(params(s)), { color: 'gold' })));
        case 'click': { const c = cellAt(s.pick, convC(s.f, 1, false)); return tex.block(tex.eq(tex.text('marker on'), c ? H('C<sub>' + (c[0] + 1) + ',' + (c[1] + 1) + '</sub> = ' + fmt(convC(s.f, 1, false).C[c[0]][c[1]])) : tex.text('—'))); }
        default: return '';
      }
    },

    steps: [
      { title: 'A filter slides', duration: 4.5,
        narration: 'Eight samples: flat, a step up, a step down, a spike. Slide the three-tap <b>filter</b> f = [1, 0, −1] along them. At each position the output is the dot product (p1-06) of the window with the taps, unflipped (ML’s <b>cross-correlation</b>): x<sub>i</sub> − x<sub>i+2</sub>. Edges light up, flat parts give zero.',
        setup(s) { s.scene = 'sig'; s.panel = 'slide'; s.view = '2d'; s.cam = camMain(); s.x = X0.slice(); s.f = EDGE.slice(); s.stride2 = false; s.pad = false; s.untie = 0; s.showC = false; s.fillRows = 0; s.fillY = 0; s.pos = 0; s.showTiles = false; s.rowHi = -1; s.band = -1; s.flip = 0; s.pull = 0; s.compact = 0; s.padFade = 0; s.cross = false; s.toggles = false; },
        at(s, t) { s.pos = 5 * seg(t, 0.05, 0.9, 'linear'); s.fillY = Math.floor(s.pos + 1e-9) + 1; } },

      { title: 'The same thing as a matrix', duration: 4.5,
        narration: 'Each output is a row of numbers dotted with the whole input: the taps sitting at a shifted position, zeros elsewhere. Stack the six rows and you have a 6×8 <b>banded matrix</b> C, the filter marching down a diagonal band. The slide <i>is</i> y = Cx.',
        setup(s) { s.showC = true; s.fillRows = 0; s.fillY = 0; s.pos = 0; s.showTiles = false; },
        at(s, t) { s.pos = 5 * seg(t, 0, 0.8, 'linear'); s.fillRows = Math.floor(s.pos + 1e-9) + 1; s.fillY = s.fillRows; s.showTiles = t > 0.85; s.band = t > 0.85 ? 9 : -1; } },

      { title: 'Row view', duration: 3,
        narration: 'Read it with p2-02’s row view: y₄ is row 4 of C dotted with x. Row 4 is the filter parked at position 4. One matrix–vector product computes every window at once: the batch idea of p6-01, along the signal instead of across samples.',
        setup(s) { s.panel = 'row'; s.showC = true; s.fillRows = 99; s.fillY = 99; s.band = -1; s.pos = 3; s.rowHi = 3; s.pull = 0; },
        at(s, t) { s.pull = seg(t, 0.3, 0.85); } },

      { title: 'Shared weights on the diagonals', duration: 3.5,
        narration: 'Look along the diagonals: every cell on the first band is 1, every cell on the third is −1, everything off the band is 0. Three numbers describe all eighteen band cells. Local (zeros far from the band) and shared (same number along a diagonal): that is <b>weight sharing</b>, and that is a convolution.',
        setup(s) { s.panel = 'shared'; s.rowHi = -1; s.pull = 0; s.pos = -1; s.band = -1; },
        at(s, t) { s.band = t < 0.9 ? Math.min(2, Math.floor(t / 0.3)) : 9; } },

      { title: 'Change the taps, keep the structure', duration: 3.5,
        narration: 'Swap in a blur, f = [1, 2, 1]/4. The band stays; its numbers change; the output smooths instead of detecting edges. The matrix shape (where the zeros are) comes from sliding; the numbers come from the taps.',
        setup(s) { s.panel = 'blur'; s.band = -1; s._f0 = s.f.slice(); },
        at(s, t) { s.f = LA.lerpV(s._f0, BLUR, seg(t, 0.1, 0.8)); } },

      { title: 'Convolution is not outside linear algebra', duration: 4.5, misconception: true,
        narration: 'It is a matrix, so everything applies: rank, transpose, gradients as outer products (p5-06). Untie the weights and each cell becomes free: still 6×8, but dense, with 48 parameters instead of 3. A fully connected layer is a convolution that forgot to share.',
        setup(s) { s.panel = 'untie'; s.f = EDGE.slice(); s.untie = 0; s.cross = true; },
        at(s, t) { s.untie = seg(t, 0.25, 0.8); } },

      { title: 'The transpose runs it backward', duration: 4.5,
        narration: 'Backprop (p5-06) pulls gradients through Cᵀ, an 8×6 matrix: each output cell spreads back to the three inputs it came from. Frameworks call this a <b>transposed convolution</b> and use it for upsampling. Nothing new: p2-10’s transpose.',
        setup(s) { s.panel = 'transpose'; s.untie = 0; s.cross = false; s.flip = 0; s.pull = 0; s.pos = -1; },
        at(s, t) { s.flip = seg(t, 0.05, 0.5); s.pull = seg(t, 0.6, 0.95); } },

      { title: 'Stride and padding', duration: 5,
        narration: 'A <b>stride</b> of 2 keeps every other window: rows 2, 4, 6 of C vanish and the output halves. <b>Padding</b> adds zeros at both ends of x: two gray columns that multiply zeros, and C grows to 8×8 so the output keeps the input’s length. Both are edits to the matrix.',
        setup(s) { s.panel = 'stride'; s.flip = 0; s.pull = 0; s.compact = 0; s.padFade = 0; s.stride2 = false; s.pad = false; },
        at(s, t) {
          s.compact = seg(t, 0.05, 0.4);
          if (t < 0.5) { s.stride2 = false; s.pad = false; s.padFade = 0; }
          else { s.compact = 0; s.stride2 = false; s.pad = true; s.padFade = seg(t, 0.5, 0.85); }
        } },

      { title: 'A 2D image is a long vector', duration: 6,
        narration: 'A 5×5 image flattened row by row (p1-07) is a 25-vector. A 3×3 kernel sliding over it becomes a 9×25 matrix: each row holds the three kernel rows, five columns (one image row) apart, shifted as the window moves. The 9 outputs reshape into a 3×3 edge map.',
        setup(s) { s.scene = 'img'; s.panel = 'img'; s.cam = camImg(); s.pad = false; s.stride2 = false; s.padFade = 0; s.unroll = 0; s.fill2d = 0; s.reshape = 0; },
        at(s, t) { s.unroll = seg(t, 0, 0.28); s.fill2d = seg(t, 0.3, 0.8, 'linear'); s.reshape = seg(t, 0.83, 1); } },

      { title: 'Why ML cares: few weights, huge matrix', duration: 4, mlTie: true,
        narration: 'A 3×3 conv from 64 to 128 channels has 73,728 weights, yet on a 224×224 input it acts as a matrix with millions of rows and columns that no one ever forms. Its gradient is a sum of outer products over every window position (p5-06). The picture stops at 3. The formula does not.',
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.cam = camMain(); },
        at(s, t) { } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'A convolution is a matrix with two constraints: zeros away from a band (local) and equal numbers along each diagonal (shared). Stride drops rows, padding adds columns, the transpose runs it backward, and an image kernel is the same pattern, one band per kernel row, five columns apart.',
        cards: ['<b>Slide = banded matrix.</b> Each window position is one row of C; y = Cx computes them all.', '<b>Shared weights = equal diagonals.</b> Stride drops rows, padding adds columns, Cᵀ runs it backward.', '<b>Images: 9×25 for a 3×3 kernel on 5×5.</b> Flatten, multiply, reshape.'],
        setup(s) { s.scene = 'recap'; s.panel = 'none'; s.cam = camMain(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Edit the three taps, switch the signal, and toggle stride, padding and untied weights. The matrix C, the output bars and the parameter count update live.</p>',
      setup(s) { s.scene = 'sig'; s.panel = 'play'; s.cam = camMain(); s.x = X0.slice(); s.f = EDGE.slice(); s.showC = true; s.fillRows = 99; s.fillY = 99; s.pos = -1; s.band = -1; s.rowHi = -1; s.flip = 0; s.pull = 0; s.compact = 0; s.padFade = 1; s.toggles = true; },
      controls: [
        { type: 'matrix', bind: 'f', label: 'Filter taps f', colors: 'gold', step: 0.05, min: -2, max: 2 },
        { type: 'preset', label: 'Filter', items: [
          { label: 'Edge detector', values: { f: [1, 0, -1] } },
          { label: 'Blur', values: { f: [0.25, 0.5, 0.25] } },
          { label: 'Identity', values: { f: [0, 1, 0] } },
        ] },
        { type: 'preset', label: 'Signal', items: [
          { label: 'edge + spike', apply(s) { s.x = X0.slice(); s.twoD = false; } },
          { label: 'ramp', apply(s) { s.x = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]; s.twoD = false; } },
          { label: 'square wave', apply(s) { s.x = [1, 1, 0, 0, 1, 1, 0, 0]; s.twoD = false; } },
          { label: '2D image view', apply(s) { s.twoD = true; } },
        ] },
        { type: 'toggle', bind: 'stride2', label: 'Stride 2' },
        { type: 'toggle', bind: 'pad', label: '‘Same’ padding' },
        { type: 'toggle', bind: 'untie', label: 'Untie weights' },
      ],
    },

    checks: [
      { kind: 'entry',
        prompt: 'Type three taps so that the output is the forward difference x<sub>i+1</sub> − x<sub>i</sub>. C and y follow your taps live.',
        bind: 'f', answer: [-1, 1, 0], tol: LA.TOL.entry, label: 'f', colors: 'gold',
        hint: 'Tap 1 multiplies x<sub>i</sub>, tap 2 multiplies x<sub>i+1</sub>.',
        setup(s) { s.scene = 'sig'; s.panel = 'play'; s.x = X0.slice(); s.f = [0, 0, 0]; s.stride2 = false; s.pad = false; s.untie = 0; s.showC = true; s.fillRows = 99; s.fillY = 99; s.pos = -1; s.band = -1; s.rowHi = -1; s.toggles = false; s.padFade = 1; } },
      { kind: 'manipulate',
        prompt: 'Drag the gold marker onto the cell of C that multiplies input x₅ in output y₃.',
        setup(s) { s.scene = 'sig'; s.panel = 'click'; s.x = X0.slice(); s.f = EDGE.slice(); s.stride2 = false; s.pad = false; s.untie = 0; s.showC = true; s.fillRows = 99; s.fillY = 99; s.pos = -1; s.band = -1; s.rowHi = -1; s.toggles = false; s.padFade = 1; s.pick = [XC(0), YC - 0.5]; },
        judge(s) { const c = cellAt(s.pick, convC(s.f, 1, false)); const ok = !!c && c[0] === 2 && c[1] === 4; const d = Math.hypot(s.pick[0] - XC(4), s.pick[1] - (YC - 2.5)); return { ok, progress: ok ? 1 : Math.max(0, 1 - d / 6), hint: c && c[0] === 4 && c[1] === 2 ? 'That is row 5, column 3. Row = output index; column = input index.' : undefined }; },
        hint: 'Row = output index; column = input index.',
        solution(s) { s.pick = [XC(4), YC - 2.5]; } },
      { kind: 'manipulate',
        prompt: 'Use the two switches at the top right of the canvas to give C the shape 4×8.',
        setup(s) { s.scene = 'sig'; s.panel = 'play'; s.x = X0.slice(); s.f = EDGE.slice(); s.stride2 = false; s.pad = false; s.untie = 0; s.showC = true; s.fillRows = 99; s.fillY = 99; s.pos = -1; s.band = -1; s.rowHi = -1; s.toggles = true; s.padFade = 1; },
        judge(s) { const R = convC(s.f, stride(s), s.pad).R; return { ok: R === 4, progress: R === 4 ? 1 : (R === 3 || R === 8 ? 0.6 : 0.2), hint: R === 3 ? 'Stride 2 alone gives 3 rows. Pad first so there are 8 window positions.' : R === 8 ? 'Padding alone gives 8 rows. Now keep every other one.' : undefined }; },
        hint: 'Stride 2 on a padded 8-sample signal gives ⌈8/2⌉ outputs.',
        solution(s) { s.stride2 = true; s.pad = true; } },
    ],
  });

  /* ---------------- helpers ---------------- */
  function cellRect(cx, cy, w, h) { return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]; }
  function cell(g, cx, cy, v, o) {
    o = o || {};
    const w = o.w || 0.94, h = o.h || 0.94;
    g.polygon(cellRect(cx, cy, w, h), { fill: o.fill || div(v), alpha: o.alpha === undefined ? 1 : o.alpha, stroke: o.stroke || null, width: o.sw || 1 });
    if (o.text !== false && (o.alpha === undefined || o.alpha > 0.35)) g.label(o.text || fmt(v), [cx, cy], { size: o.size || 12, color: o.textColor || (Math.abs(v) < 1e-9 ? 'muted' : 'text'), bg: false, font: 'ui' });
  }
  function cellAt(p, cv) {   // which cell of the default C (no padding) lies under world point p
    const j = Math.floor(p[0] + 4), r = Math.floor(YC - p[1]);
    return j >= 0 && j < 8 && r >= 0 && r < cv.R ? [r, j] : null;
  }
  function xBars(g, s, o) {
    o = o || {};
    const x = s.x;
    g.line([XC(-0.5) - 0.1, YX], [XC(7.5) + 0.1, YX], { color: 'gridAxis', width: 1.2 });
    for (let j = 0; j < 8; j++) {
      const v = x[j], hgt = v * BAR, c = o.gold && o.gold[j] !== undefined ? 'gold' : 'v';
      if (Math.abs(v) > 1e-9) g.polygon([[XC(j) - 0.32, YX], [XC(j) + 0.32, YX], [XC(j) + 0.32, YX + hgt], [XC(j) - 0.32, YX + hgt]], { fill: 'v', alpha: o.dim ? 0.35 : 0.85, stroke: 'v' });
      g.label('x' + sub(j), [XC(j), YX], { anchor: 'below', dy: 3, size: 11, color: 'muted', font: 'ui' });
      if (Math.abs(v) > 1e-9) g.label(fmt(v), [XC(j), YX + hgt], { anchor: 'above', dy: -2, size: 11, color: 'v', font: 'ui' });
      if (o.gold && o.gold[j] !== undefined) {
        const gv = o.gold[j];
        g.polygon([[XC(j) - 0.18, YX], [XC(j) + 0.18, YX], [XC(j) + 0.18, YX + gv * BAR], [XC(j) - 0.18, YX + gv * BAR]], { fill: 'gold', alpha: 0.95, stroke: 'gold' });
        g.label(fmt(gv), [XC(j) + 0.45, YX + gv * BAR * 0.5 + (gv === 0 ? 0.25 : 0)], { size: 11.5, color: 'gold', font: 'ui', anchor: 'left' });
        void c;
      }
    }
    if (o.ghostPad) [-1, 8].forEach(j => { g.polygon(cellRect(XC(j), YX + 0.25, 0.64, 0.5), { fill: 'ghost', alpha: 0.6 * o.ghostPad, stroke: 'muted' }); g.label('0', [XC(j), YX + 0.25], { size: 11, color: 'muted', bg: false, font: 'ui' }); });
    g.label('x', [XC(-1) - 0.8, YX + 0.6], { color: 'v', size: 16 });
  }
  function yCol(g, y, rowsY, o) {
    o = o || {};
    const n = y.length;
    g.label('y', [YCOL, YC + 0.45], { color: 'b', size: 16 });
    for (let r = 0; r < n; r++) {
      const cy = rowsY[r], a = o.alpha ? o.alpha[r] : 1;
      if (a <= 0.01) continue;
      const filled = o.fill === undefined || r < o.fill;
      g.polygon(cellRect(YCOL, cy, 0.94, 0.94), { fill: 'panel2', alpha: a, stroke: o.hi === r ? 'gold' : 'line', width: o.hi === r ? 2 : 1 });
      if (!filled) continue;
      const v = y[r], wv = LA.clamp(v, -1, 1) * 0.45;
      g.polygon([[YCOL, cy - 0.28], [YCOL + wv, cy - 0.28], [YCOL + wv, cy + 0.28], [YCOL, cy + 0.28]], { fill: 'b', alpha: 0.5 * a, stroke: null });
      g.label(fmt(v), [YCOL, cy], { size: 12, color: 'text', bg: false, font: 'ui', alpha: a });
      g.label('y' + sub(r), [YCOL + 0.75, cy], { size: 11, color: 'muted', font: 'ui', anchor: 'left', alpha: a });
    }
  }
  function tiles(g, R, cy) { g.label('(' + R + '×8)(8×1) → (' + R + '×1)', [0, cy], { size: 14, color: 'hud', font: 'ui' }); }
  function toggles(g, s) {
    const T = (id, label, on, y, set) => {
      const xa = 5.0, xb = 6.4;
      g.label(label, [xa - 0.2, y + 0.42], { anchor: 'left', size: 11.5, color: 'muted', font: 'ui' });
      g.line([xa, y], [xb, y], { color: on ? 'gold' : 'ghostAxis', width: 4 });
      g.label('off', [xa - 0.15, y], { anchor: 'right', size: 10.5, color: 'muted', font: 'ui' }); g.label('on', [xb + 0.15, y], { anchor: 'left', size: 10.5, color: 'muted', font: 'ui' });
      const hp = [on ? xb : xa, y];
      g.point(hp, { r: 6, color: on ? 'gold' : 'muted' });
      g.handle(id, hp, (p, st) => set(st, p[0] > (xa + xb) / 2), { color: 'gold' });
    };
    T('stride', 'stride 2', s.stride2, 7.0, (st, v) => { st.stride2 = v; });
    T('pad', '‘same’ padding', s.pad, 5.95, (st, v) => { st.pad = v; });
  }

  /* ---------------- the 1D scene ---------------- */
  function renderSig(g, s, ctx) {
    const cv = convC(s.f, stride(s), s.pad, s.untie);
    const y = outY(cv, s.x);
    // window of the current slide position
    const win = s.pos >= 0 ? Math.min(5, Math.floor(s.pos + 1e-9)) : -1;
    const gold = {};
    if (s.pull > 0 && s.panel === 'transpose') { gold[2] = EDGE[0] * s.pull; gold[3] = 0; gold[4] = EDGE[2] * s.pull; }
    xBars(g, s, { ghostPad: s.pad ? s.padFade : 0, gold: s.panel === 'transpose' && s.pull > 0 ? gold : null, dim: s.panel === 'transpose' && s.pull > 0 });
    if (win >= 0 && s.panel !== 'transpose') {
      const wx = XC(s.pos + 1);
      g.polygon(cellRect(wx, YX + 0.85, 3, 2.1), { fill: 'gold', alpha: 0.1, stroke: 'gold', width: 1.4 });
      if (!s.showC) EDGE.forEach((fk, k) => cell(g, XC(s.pos + k), YX - 0.75, s.f[k], { w: 0.7, h: 0.5, size: 11 }));
    }
    const R = cv.R;
    // rows' vertical positions (stride compaction animates rows upward)
    const rowY = r => YC - 0.5 - r;
    if (s.panel === 'transpose') { renderTranspose(g, s, cv, y); return; }
    if (s.showC) {
      const rows = [];
      if (s.compact > 0) {   // stride 2 being applied to the 6×8 matrix: odd rows fade, even rows close up
        for (let r = 0; r < R; r++) { const keep = r % 2 === 0; rows.push({ r, y: lerp(rowY(r), rowY(r / 2), keep ? s.compact : 0), a: keep ? 1 : 1 - s.compact }); }
      } else for (let r = 0; r < R; r++) rows.push({ r, y: rowY(r), a: s.pad ? (r === 0 || r === R - 1 ? s.padFade : 1) : 1 });
      rows.slice().sort((p, q) => p.a - q.a).forEach(({ r, y: cy, a }) => {   // fading rows first, kept rows on top
        if (r >= s.fillRows) { for (let c = 0; c < cv.L; c++) g.polygon(cellRect(XC(c - cv.P), cy, 0.94, 0.94), { fill: 'panel2', alpha: 0.6, stroke: 'line' }); return; }
        for (let c = 0; c < cv.L; c++) {
          const ghost = cv.P && (c === 0 || c === cv.L - 1);
          const v = cv.C[r][c], k = c - r * stride(s);
          const inBand = !s.untie && k >= 0 && k <= 2;
          const pulse = s.band >= 0 && inBand && (s.band === 9 || s.band === k);
          const hiRow = s.rowHi === r;
          cell(g, XC(c - cv.P), cy, v, { alpha: a * (ghost ? 0.35 * s.padFade : 1), stroke: pulse ? 'gold' : hiRow ? 'v' : 'line', sw: pulse || hiRow ? 2 : 1, size: s.untie > 0.05 ? 10 : 12, text: s.untie > 0.05 ? fmt.fixed(v, 1) : undefined });
        }
        g.label(String(r + 1), [XC(-cv.P) - 0.6, cy], { anchor: 'right', size: 10.5, color: s.rowHi === r ? 'v' : 'muted', font: 'ui', alpha: a });
      });
      // y column follows the rows
      yCol(g, y, rows.map(o => o.y), { fill: Math.min(s.fillY, R), hi: s.rowHi, alpha: rows.map(o => o.a) });
      const shownR = s.compact > 0.95 ? Math.ceil(R / 2) : R;
      g.label('C ∈ ℝ' + supNum(shownR) + 'ˣ⁸', [XC(-cv.P) - 0.4, YC + 0.45], { anchor: 'left', size: 14, color: 'text', font: 'ui' });
      if (s.showTiles || s.panel === 'stride' || s.panel === 'play' || s.panel === 'click') tiles(g, shownR, rowY(Math.max(R, 6)) - 0.35);
      if (s.band === 9 && s.panel === 'matrix') g.polyline([[XC(-0.5), YC], [XC(2.5), YC], [XC(7.5), YC - 6], [XC(4.5), YC - 6]], { color: 'gold', width: 1.6, closed: true, dashed: true });
      if (s.panel === 'shared') g.label('3 parameters · 18 band cells · 48 cells', [0.3, rowY(6) - 1.15], { size: 13.5, color: 'gold', font: 'ui' });
      if (s.panel === 'untie' || s.panel === 'play') g.label('parameters: ' + params(s), [0.3, rowY(Math.max(R, 6)) - 1.15], { size: 13.5, color: s.untie > 0.5 ? 'bad' : 'gold', font: 'ui' });
      if (s.cross) { g.label('“a special op, not a matrix”', [-2.4, rowY(6) - 1.95], { anchor: 'left', size: 14, color: 'bad', font: 'ui' }); g.marker.cross([-2.8, rowY(6) - 1.95], { size: 20 }); }
      // row view: the dot product of row 4 and x lands in y₄ as a magenta shadow bar
      if (s.rowHi >= 0 && s.pull > 0) {
        const r = s.rowHi, cy = rowY(r);
        for (let j = 0; j < 8; j++) if (Math.abs(cv.C[r][j + cv.P]) > 1e-9) g.line([XC(j), YX - 0.3], [XC(j), cy + 0.5], { color: 'v', dashed: true, width: 1.2, alpha: 0.7 });
        const v = y[r] * s.pull;
        g.polygon([[YCOL - 0.47, cy - 0.47], [YCOL - 0.47 + 0.94 * Math.min(1, Math.abs(v)), cy - 0.47], [YCOL - 0.47 + 0.94 * Math.min(1, Math.abs(v)), cy - 0.3], [YCOL - 0.47, cy - 0.3]], { fill: 'mag', alpha: 0.9 });
      }
      // the gold marker (check 2) and its cell
      if (s.panel === 'click') {
        const c = cellAt(s.pick, cv);
        if (c) g.polyline(cellRect(XC(c[1]), rowY(c[0]), 0.98, 0.98), { color: 'gold', width: 2.6, closed: true });
        g.point(s.pick, { r: 6, color: 'gold', stroke: 'white', strokeWidth: 1.5 });
        g.handle('pick', s.pick, (p, st) => { st.pick = [LA.clamp(p[0], -5, 7), LA.clamp(p[1], -4, 7)]; }, { color: 'gold' });
      }
    } else {
      yCol(g, y, y.map((_, r) => rowY(r)), { fill: s.fillY });
      if (win >= 0) { const cy = rowY(win); g.vector([XC(s.pos + 1), YX - 1.1], [YCOL - 0.6, cy], { color: 'gold', width: 1.6, head: 9, alpha: 0.8 }); }
    }
    if (s.toggles) toggles(g, s);
  }
  function supNum(n) { return String(n).split('').map(d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join(''); }

  function renderTranspose(g, s, cv, y) {
    const u = s.flip, R = 6;
    const C = realC(convC(EDGE, 1, false));
    for (let r = 0; r < R; r++) for (let j = 0; j < 8; j++) {
      const a = [XC(j), YC - 0.5 - r], b = [XC(r), YC - 0.5 - j];
      const p = LA.lerpV(a, b, u);
      const lit = s.pull > 0 && r === 2;            // column 3 of Cᵀ = row 3 of C
      cell(g, p[0], p[1], C[r][j], { stroke: lit ? 'gold' : 'line', sw: lit ? 2 : 1 });
    }
    g.label(u < 0.5 ? 'C ∈ ℝ⁶ˣ⁸' : 'Cᵀ ∈ ℝ⁸ˣ⁶', [XC(0) - 0.4, YC + 0.45], { anchor: 'left', size: 14, color: 'text', font: 'ui' });
    // ∂L/∂y = e₃ on the right, pulled back to the inputs
    if (u > 0.95) {
      g.label('∂L/∂y', [YCOL - 1.2, YC + 0.45], { anchor: 'left', size: 13, color: 'gold', font: 'ui' });
      for (let r = 0; r < R; r++) cell(g, YCOL - 0.6, YC - 0.5 - r, r === 2 ? 1 : 0, { fill: r === 2 ? 'gold' : 'panel2', textColor: r === 2 ? 'bg' : 'muted' });
      if (s.pull > 0) {
        [2, 3, 4].forEach(j => g.line([XC(2) + 0.5, YC - 0.5 - j], [YCOL - 1.1, YC - 2.5], { color: 'gold', width: 1, alpha: 0.5 * s.pull, dashed: true }));
        g.label('Cᵀe₃ lights x₃, x₄, x₅ with 1, 0, −1', [0.5, YC - 9.0], { size: 13.5, color: 'gold', font: 'ui' });
      }
    }
  }

  /* ---------------- the 2D image scene ---------------- */
  function renderImg(g, s) {
    const u1 = s.unroll, u2 = s.fill2d, u3 = s.reshape;
    const ci = 0.62, I0 = [-6.0, 6.6];                         // image cells, top-left corner
    const cs = 0.46, S0 = [-5.75, 2.1];                        // strip / matrix cell size and left x, strip y
    const M0y = 1.3;                                           // top of C2d
    const imgPos = (i, j) => [I0[0] + (j + 0.5) * ci, I0[1] - (i + 0.5) * ci];
    const stripPos = k => [S0[0] + (k + 0.5) * cs, S0[1]];
    // the image (stays) and its flying copy
    g.label('5×5 image (synthetic)', [I0[0], I0[1] + 0.35], { anchor: 'left', size: 12, color: 'muted', font: 'ui' });
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { const p = imgPos(i, j); g.polygon(cellRect(p[0], p[1], ci * 0.96, ci * 0.96), { fill: IMG[i][j] ? 'white' : 'panel2', alpha: 0.9, stroke: 'line' }); }
    const done = Math.floor(9 * u2 + 1e-9);            // outputs computed so far
    const wk = Math.min(8, done);                      // current window (row-major over the 3×3 positions)
    if (u2 > 0 && u2 < 1) { const wi = Math.floor(wk / 3), wj = wk % 3; const p = imgPos(wi + 1, wj + 1); g.polygon(cellRect(p[0], p[1], 3 * ci, 3 * ci), { fill: 'gold', alpha: 0.12, stroke: 'gold', width: 2 }); }
    for (let k = 0; k < 25; k++) {
      const i = Math.floor(k / 5), j = k % 5, p = LA.lerpV(imgPos(i, j), stripPos(k), u1);
      if (u1 > 0) g.polygon(cellRect(p[0], p[1], lerp(ci, cs, u1) * 0.94, lerp(ci, cs, u1) * 0.94), { fill: FLAT[k] ? 'white' : 'panel2', alpha: 0.95, stroke: 'line' });
    }
    if (u1 > 0.9) g.label('x: the image as a 25-vector (row by row)', [S0[0], S0[1] + 0.5], { anchor: 'left', size: 12, color: 'v', font: 'ui' });
    // C2d, 9 × 25
    if (u2 > 0) {
      for (let r = 0; r < 9; r++) {
        const cy = M0y - (r + 0.5) * cs;
        for (let c = 0; c < 25; c++) {
          const filled = r < done || u2 >= 1;
          const v = C2D[r][c];
          g.polygon(cellRect(S0[0] + (c + 0.5) * cs, cy, cs * 0.92, cs * 0.92), { fill: filled ? div(v) : 'panel2', alpha: filled ? 1 : 0.5, stroke: null });
        }
      }
      g.label('C₂d ∈ ℝ⁹ˣ²⁵: one row per window, a kernel row every 5 columns', [S0[0], M0y - 9 * cs - 0.35], { anchor: 'left', size: 12, color: 'muted', font: 'ui' });
      // outputs column (then reshaped into 3×3)
      const oc = 0.62, colX = S0[0] + 25 * cs + 0.75, M3 = [2.6, 6.6];
      for (let r = 0; r < 9; r++) {
        if (r >= done && u2 < 1) continue;
        const a = [colX, M0y - (r + 0.5) * cs], b = [M3[0] + (r % 3 + 0.5) * oc, M3[1] - (Math.floor(r / 3) + 0.5) * oc];
        const p = LA.lerpV(a, b, u3), w = lerp(cs, oc, u3);
        cell(g, p[0], p[1], Y2D[r] / 2, { w: w * 0.94, h: w * 0.94, text: fmt(Y2D[r]), size: u3 > 0.5 ? 11.5 : 9.5 });
      }
      if (u3 > 0.9) g.label('3×3 edge map: vertical edges', [M3[0], M3[1] + 0.35], { anchor: 'left', size: 12, color: 'b', font: 'ui' });
      else g.label('y', [colX, M0y + 0.3], { size: 14, color: 'b' });
    }
    g.label('Sobel-x kernel [[1, 0, −1], [2, 0, −2], [1, 0, −1]]', [-2.5, 4.05], { anchor: 'left', size: 12, color: 'gold', font: 'ui' });
  }

  /* ---------------- ML tie-in and recap ---------------- */
  function renderML(g, s) {
    const W = g.size.w, H = g.size.h;
    g.viewport({ x: 0.03, y: 0.04, w: 0.94, h: 0.7 }, { view: '2d', cam: { center: [16, -12], halfH: Math.max(14, 17.5 * (0.7 * H) / (0.94 * W)) }, bg: 'panel' }, (g2, r) => {
      g2.heat2d(BIG, { x0: 0, x1: 32, y0: -24, y1: 0 }, { range: [-2, 2], ramp: t => LA.color.diverge(2 * t - 1), smooth: false, alpha: 1 });
      for (let b = 0; b <= 4; b++) g2.line([b * 8, 0], [b * 8, -24], { color: 'line', width: 1 });
      for (let a = 0; a <= 3; a++) g2.line([0, -a * 8], [32, -a * 8], { color: 'line', width: 1 });
      g2.label('3 output × 4 input channels: every block is banded and shared', [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13, color: 'text', font: 'ui', bold: true });
      g2.label('a 24×32 sketch; the real one is ~6·10⁶ × 3·10⁶', [r.x + 12, r.y + r.h - 16], { screen: true, anchor: 'left', size: 12, color: 'muted', font: 'ui' });
    });
    g.hud('73,728 weights · ~6·10⁶ × 3·10⁶ matrix, never formed', 0.05 * W, 0.8 * H, { size: 14, color: 'gold' });
    g.hud('gradient: ∂L/∂f = Σ over positions of δᵢ · window(x)ᵢ', 0.05 * W, 0.8 * H + 26, { size: 13, color: 'hud' });
    g.hud('The picture stops at 3. The formula does not.', 0.05 * W, 0.8 * H + 52, { size: 14, color: 'gold' });
    g.hud('see p6-08', 0.95 * W, 0.8 * H + 52, { size: 13, color: 'muted', anchor: 'right' });
  }

  function renderRecap(g, s) {
    const card = (i, cam, fn, text) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, { view: '2d', cam, bg: 'panel' }, (g2, r) => { fn(g2, r); g2.label(text, [r.x + 12, r.y + 16], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true }); });
    const tied = realC(convC(EDGE, 1, false));
    card(0, { center: [0.6, 5.0], halfH: 1.9 }, (g2) => {
      const st = { x: X0 };
      xBars(g2, st, {});
      g2.polygon(cellRect(XC(3), YX + 0.85, 3, 2.1), { fill: 'gold', alpha: 0.12, stroke: 'gold', width: 1.4 });
      EDGE.forEach((fk, k) => cell(g2, XC(2 + k), YX - 0.75, fk, { w: 0.7, h: 0.5, size: 11 }));
    }, '1 · slide = one row per window');
    card(1, { center: [0.3, 0.75], halfH: 4.1 }, (g2) => {
      for (let r = 0; r < 6; r++) for (let j = 0; j < 8; j++) cell(g2, XC(j), YC - 0.5 - r, tied[r][j], { size: 10.5, stroke: j - r >= 0 && j - r <= 2 ? 'gold' : 'line' });
      g2.label('equal diagonals', [XC(7.5) + 0.4, YC - 3], { anchor: 'left', size: 12.5, color: 'gold', font: 'ui' });
    }, '2 · banded, shared: C ∈ ℝ⁶ˣ⁸');
    card(2, { center: [6, -2.3], halfH: 3.0 }, (g2) => {
      const cs = 0.42;
      for (let r = 0; r < 9; r++) for (let c = 0; c < 25; c++) g2.polygon(cellRect((c + 0.5) * cs, -(r + 0.5) * cs, cs * 0.9, cs * 0.9), { fill: div(C2D[r][c]), alpha: 1 });
      for (let r = 0; r < 9; r++) cell(g2, 25 * cs + 1.4 + (r % 3) * 0.7, -0.6 - Math.floor(r / 3) * 0.7, Y2D[r] / 2, { w: 0.66, h: 0.66, text: fmt(Y2D[r]), size: 10 });
    }, '3 · images: 9×25 for a 3×3 kernel on 5×5 → 3×3 edge map');
  }
})();
