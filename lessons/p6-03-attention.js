/* p6-03 — Attention as dot products, softmax, weighted sum. Scores via LA.num.softmax(scores, T). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, C = LA.color, lerp = LA.lerp, lerpV = LA.lerpV, clamp01 = LA.clamp01;
  const PX = 0.4;
  const TOK = ['the₁', 'cat', 'sat', 'on', 'the₂', 'mat'];
  const X0 = [[0.3, 0.1, 0.2], [1.0, 0.1, 0.0], [0.1, 1.0, 0.1], [0.0, 0.1, 1.0], [0.3, -0.1, 0.2], [0.9, 0.0, 0.3]];
  const WQ = [[0.6, 3.6, 0.0], [0.2, 0.0, 0.2], [0.0, -2.4, 0.0]];
  const WQ2 = [[0.0, 0.0, 2.0], [0.2, 0.0, 0.0], [0.0, 0.0, 1.0]];
  const WK = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const WV = [[0.8, 0, 0.2], [0, 0.8, 0], [0.2, 0, 0.8]];
  const DEXP = [3, 16, 64, 256, 512];
  const QS = 0.5;                                  // the query is drawn at half length so it fits the scene
  const qv = '<span class="c-v">q</span>', kv = '<span class="c-teal">k</span>', vv = '<span class="c-w">v</span>', ov = '<span class="c-b">o</span>';

  // ---------- heavy data, computed once ----------
  let CACHE = null;
  function heavy() {
    if (CACHE) return CACHE;
    const raw = [], scaled = [];
    DEXP.forEach(d => {
      const U = LA.rng.tokenVectors(603, 800, d), sq = Math.sqrt(d);
      const vals = []; for (let p = 0; p < 400; p++) vals.push(N.dot(U[2 * p], U[2 * p + 1]) * d);   // entries of unit spread: q = √d·u
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length);
      raw.push(Math.round(sd * 100) / 100); scaled.push(Math.round(sd / sq * 100) / 100);
    });
    const qk = LA.rng.tokenVectors(604, 2, 64).map(v => v.map(x => Math.round(x * 8 * 100) / 100));
    const Q12 = LA.rng.tokenVectors(604, 12, 16), K12 = LA.rng.tokenVectors(605, 12, 16);
    const A12 = Q12.map(q => N.softmax(K12.map(k => 4 * N.dot(q, k)), 1).map(x => Math.round(x * 1000) / 1000));
    CACHE = { raw, scaled, qk, A12 };
    return CACHE;
  }

  // ---------- the attention computation (live) ----------
  function attn(s) {
    const Wq = s.head === 'B' ? WQ2 : WQ;
    const Q = s.X.map(x => N.matVec(Wq, x)), K = s.X.map(x => N.matVec(WK, x)), V = s.X.map(x => N.matVec(WV, x));
    const S = Q.map(q => K.map(k => N.dot(q, k)));
    const sc = s.scale ? Math.sqrt(3) : 1;
    const A = S.map(r => N.softmax(r.map(x => x / sc), s.T));
    const O = A.map(a => { let o = [0, 0, 0]; a.forEach((w, j) => { o = N.add(o, N.scale(V[j], w)); }); return o; });
    return { Q, K, V, S, A, O, sc };
  }

  // ---------- screen helpers ----------
  function rect(g, x, y, w, h, fill) { g.strip([0], { x, y, w, h, range: [0, 1], ramp: () => fill }); }
  function txt(g, s, x, y, o) { g.label(s, [x, y], Object.assign({ screen: true, size: 12.5, bg: false, font: 'ui', anchor: 'left', color: 'hud' }, o || {})); }
  const rampDiv = t => C.diverge(2 * t - 1), rampHeat = t => C.heat(t);
  /** heat table; o: cw, ch, range, ramp, rowLabels, colLabels, hiRow, cellsShown(i, j) → bool, digits, title */
  function table(g, M, x, y, o) {
    o = o || {};
    const cw = o.cw || 28, ch = o.ch || 20, R = o.range || [-3, 3], ramp = o.ramp || rampDiv;
    M.forEach((row, i) => row.forEach((v, j) => {
      const shown = !o.shown || o.shown(i, j);
      const t = (v - R[0]) / (R[1] - R[0]);
      rect(g, x + j * cw, y + i * ch, cw - 1, ch - 1, shown ? ramp(clamp01(t)) : C.panel2);
      if (shown && o.values !== false) txt(g, LA.fmt(v), x + (j + 0.5) * cw, y + (i + 0.5) * ch, { anchor: 'center', size: o.size || 9.5, color: o.textColor ? o.textColor(t) : 'text' });
    }));
    if (o.rowLabels) o.rowLabels.forEach((l, i) => txt(g, l, x - 4, y + (i + 0.5) * ch, { anchor: 'right', size: 10, color: o.hiRow === i ? 'v' : 'muted' }));
    if (o.colLabels) o.colLabels.forEach((l, j) => txt(g, l, x + (j + 0.5) * cw, y - 8, { anchor: 'center', size: 9.5, color: o.hiCol === j ? 'teal' : 'muted' }));
    if (o.hiRow !== undefined && o.hiRow !== null && o.frame !== false) { rect(g, x - 1, y + o.hiRow * ch - 1, M[0].length * cw, 2, C.v); rect(g, x - 1, y + (o.hiRow + 1) * ch - 2, M[0].length * cw, 2, C.v); }
    if (o.title) txt(g, o.title, x, y - (o.colLabels ? 24 : 10), { size: 12, color: o.titleColor || 'hud' });
  }
  function tile(g, x, y, r, c, o) {
    o = o || {}; const k = o.k || 8, w = c * k, h = r * k;
    rect(g, x, y - h / 2, w, h, C.alpha(C.role(o.color || 'teal'), 0.35));
    txt(g, r + '×' + c, x + w / 2, y + h / 2 + 11, { anchor: 'center', size: 11, font: 'mono', color: 'text' });
    return { x, y, w, h };
  }
  /** vertical weight bars under token labels */
  function wbars(g, w, x, y, bw, h, o) {
    o = o || {};
    w.forEach((a, j) => {
      const bh = a * h;
      rect(g, x + j * bw + 3, y + h - bh, bw - 6, Math.max(1, bh), C.alpha(C.role(o.colors ? o.colors[j] : 'gold'), o.alpha === undefined ? 0.85 : o.alpha));
      if (o.values !== false) txt(g, LA.fmt(a), x + (j + 0.5) * bw, y + h - bh - 8, { anchor: 'center', size: 10, color: 'hud' });
      txt(g, o.labels ? o.labels[j] : TOK[j], x + (j + 0.5) * bw, y + h + 10, { anchor: 'center', size: 10, color: 'muted' });
    });
    rect(g, x, y + h, w.length * bw, 1, C.line);
    if (o.title) txt(g, o.title, x, y - 12, { size: 12, color: o.titleColor || 'hud' });
  }

  const cam3 = () => Object.assign(LA.cam.defaults3d(), { target: [0.6, 0.25, -0.15], dist: 6.2, yaw: -0.75, pitch: 0.4, fov: 36 });
  const camClose = () => Object.assign(LA.cam.defaults3d(), { target: [0.35, 0.15, 0.2], dist: 3.8, yaw: -0.75, pitch: 0.4, fov: 36 });

  LA.registerLesson({
    id: 'p6-03-attention',
    part: 6,
    title: 'Attention as dot products, softmax, weighted sum',
    subtitle: 'Shadows of keys on a query, softmax of the row, a tip-to-tail sum of values.',
    minutes: 14,
    prereqs: ['p6-02-embeddings-analogies', 'p1-06-dot-product', 'p3-07-gram-matrix', 'p4-04-simplex-softmax', 'p6-01-batches-shapes-tensors', 'p2-02-matrix-vector'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '3D (d = 3, 6 tokens)', realSize: 'd_k = 64–128 per head, 8–96 heads, T up to 10⁵ tokens', transfers: 'scores are dot products, softmax acts per row, outputs are weighted sums — identical in any d; only the number of arrows and cells changes' },
    cantShow: ['64 arrows per token; one 3D arrow stands for a 64-cell strip', 'causal masks and real positional encodings (we add a small positional nudge only)', 'more than two heads, and the T×T heatmap at scale'],
    terms: [
      { term: 'token', def: 'One word piece, carried as a vector.' },
      { term: 'query', def: 'q = W_q x: what a token is looking for. Keys (k = W_k x) and values (v = W_v x) are two more linear maps of the same token vector.' },
      { term: 'score', def: 'The dot product q·k: how much a query matches a key.' },
      { term: 'attention weights', def: 'Softmax of a row of scores: non-negative and summing to one.' },
      { term: 'weighted sum', def: 'Σ αⱼvⱼ: a convex combination of the value vectors.' },
      { term: 'scaling by √d', def: 'Dividing scores by √d so their spread does not grow with the dimension.' },
      { term: 'head', def: 'One set of W_q, W_k, W_v; several heads run in parallel and their outputs are stacked.' },
    ],
    atlas: ['attention', 'softmax', 'embeddings'],
    thumbnail: 7,

    state() {
      const H = heavy();
      return {
        view: '3d', cam: cam3(),
        X: LA.clone(X0), head: 'A', q: 2, T: 1, scale: true, dIdx: 2,
        sdRaw: H.raw, sdScaled: H.scaled, qk64: H.qk, A12: H.A12,
        scene: 'attn', panel: 'sentence', mathMode: 'X', u: 0,
        arrows: 'x', showQ: 0, showV: 0, shadowN: 0, fillRow: -1, fillRows: 0, divU: 1, chainU: 0, showAllO: 0,
        oneHot: 0, guessA: [0.33, 0.33, 0.34], checkMode: null, dragCat: false,
      };
    },

    render(g, s, ctx) {
      const R = attn(s);
      if (s.scene === 'attn' && s.view === '3d') {
        g.viewport({ x: PX, y: 0, w: 1 - PX, h: 1 }, { view: '3d', cam: s.cam, border: false }, (gg) => drawScene(gg, s, R, ctx));
        g.strip([0], { x: PX * g.size.w - 1, y: 0, w: 1, h: g.size.h, range: [0, 1], ramp: () => C.line });
      }
      drawPanel(g, s, R);
    },

    math(s, tex) {
      const R = attn(s), q = s.q;
      switch (s.mathMode) {
        case 'X': return tex.block(tex.eq(tex.sym('X'), tex.op('∈'), tex.text('ℝ⁶ˣ³')), tex.eq(tex.text('rows = tokens')));
        case 'maps': return tex.block(tex.eq(tex.sym('q', { color: 'v' }), tex.op('='), tex.sym('W', { sub: 'q' }), tex.sym('x'), tex.op('·'), tex.sym('k', { color: 'teal' }), tex.op('='), tex.sym('W', { sub: 'k' }), tex.sym('x'), tex.op('·'), tex.sym('v', { color: 'w' }), tex.op('='), tex.sym('W', { sub: 'v' }), tex.sym('x')), tex.eq(tex.sym('q', { color: 'v', sub: 'sat' }), tex.op('='), tex.vec(R.Q[2], { color: 'v' })));
        case 'score': return tex.block(tex.eq(tex.sym('s', { sub: 'j' }), tex.op('='), tex.sym('q', { color: 'v' }), tex.op('·'), tex.sym('k', { color: 'teal', sub: 'j' })), tex.eq(tex.text('row sat ='), tex.rowvec(R.S[2])));
        case 'S': return tex.block(tex.eq(tex.sym('S'), tex.op('='), tex.sym('Q'), tex.T(tex.sym('K')), tex.op('∈'), tex.text('ℝ⁶ˣ⁶')));
        case 'sqrt': return tex.block(tex.eq(tex.text('std(q·k) ≈ √d'), tex.op('·'), tex.text('std(q·k/√d) ≈ 1')));
        case 'soft': return tex.block(tex.eq(tex.text('α = softmax(s/√d, T)'), tex.op('·'), tex.text('Σⱼαⱼ = 1')), tex.eq(tex.text('T ='), tex.num(s.T, { color: 'gold' }), tex.text(' α ='), tex.rowvec(R.A[q], { color: 'gold' })));
        case 'onehot': return tex.block(tex.eq(tex.text('one-hot', { color: 'bad' }), tex.text(' ✕', { color: 'bad' })), tex.eq(tex.text('α ='), tex.rowvec(R.A[2], { color: 'gold' }), tex.text(' ✓', { color: 'ok' })));
        case 'sum': return tex.block(tex.eq(tex.sym('o', { color: 'b' }), tex.op('='), tex.sum(tex.text('αⱼ ') + tex.sym('v', { color: 'w', sub: 'j' }), 'j'), tex.op('='), tex.vec(R.O[q], { color: 'b' })));
        case 'AV': return tex.block(tex.eq(tex.sym('O'), tex.op('='), tex.text('softmax(QKᵀ/√d)'), tex.sym('V')), tex.eq(tex.text('(6×6)(6×3) → (6×3)')));
        case 'drag': return tex.block(tex.eq(tex.text('α(sat, cat) ='), tex.num(R.A[2][1], { color: 'gold' })), tex.eq(tex.text('cat ='), tex.vec(s.X[1], { color: 'teal' })));
        case 'headB': return tex.block(tex.eq(tex.text('head B: '), tex.sym('W', { sub: 'q' }), '′', tex.op('='), tex.mat(WQ2)), tex.eq(tex.sym('O'), tex.op('='), tex.text('[O_A O_B]'), tex.T(tex.sym('W', { sub: 'o' }))));
        case 'ml': return tex.block(tex.eq(tex.sym('S'), tex.op('∈'), tex.text('ℝᵀˣᵀ'), tex.op('·'), tex.text('O = softmax(QKᵀ/√d)V')));
        case 'play': return tex.block(tex.eq(tex.text('query: ' + TOK[q] + ' · head ' + s.head + ' · T ='), tex.num(s.T)), tex.eq(tex.text('α ='), tex.rowvec(R.A[q], { color: 'gold' })), tex.eq(tex.sym('o', { color: 'b' }), tex.op('='), tex.rowvec(R.O[q], { color: 'b' })));
        case 'temp': { const a = R.A[2]; return tex.block(tex.eq(tex.text('T ='), tex.num(s.T, { color: 'gold' })), tex.eq(tex.text('max − min ='), tex.num(Math.max(...a) - Math.min(...a), { color: 'gold' }))); }
        case 'mini': return tex.block(tex.eq(tex.text('softmax(2, 0, 0) = ?')), tex.eq(tex.text('your guess:'), tex.rowvec(s.guessA, { color: 'gold' })));
        default: return '';
      }
    },

    steps: [
      { title: 'Six tokens', duration: 3,
        narration: '“the cat sat on the mat”: six <b>token</b>s, each a 3D vector (p6-02), stacked as the rows of X (p6-01). The two “the”s differ by a small positional nudge so position can matter. The question attention answers: which tokens should “sat” look at?',
        setup(s) { Object.assign(s, { scene: 'attn', view: '3d', panel: 'sentence', mathMode: 'X', X: LA.clone(X0), head: 'A', q: 2, T: 1, scale: true, arrows: 'x', showQ: 0, showV: 0, shadowN: 0, fillRow: -1, fillRows: 0, divU: 1, chainU: 0, showAllO: 0, oneHot: 0, u: 0 }); s.cam = cam3(); },
        at(s, t) { s.u = t; LA.cam.tween(s, { yaw: -1.2 }, { yaw: -1.0 }, t); } },

      { title: 'Three maps, three roles', duration: 4,
        narration: 'Three matrices (p2-01) give every token three roles: a <b>query</b> ' + qv + ' = W_q x (“what am I looking for?”), a key ' + kv + ' = W_k x (“what do I offer?”), a value ' + vv + ' = W_v x (“what do I pass on?”). For sat we draw its query; for every token, its key.',
        setup(s) { Object.assign(s, { panel: 'maps', mathMode: 'maps', arrows: 'x', showQ: 0, showV: 0, u: 0 }); },
        at(s, t) { s.u = t; s.showQ = seg(t, 0.1, 0.5); s.arrows = t > 0.55 ? 'k' : 'x'; s.showV = seg(t, 0.7, 1); } },

      { title: 'Scores are shadows', duration: 5,
        narration: 'How well does each key match the query? Its shadow on the query (p1-06): the dot product, a <b>score</b> q·kⱼ. Six shadows, six numbers — one row of the score table. cat casts the longest shadow; on points the other way.',
        setup(s) { Object.assign(s, { panel: 'scores', mathMode: 'score', showQ: 1, showV: 0, arrows: 'k', shadowN: 0, fillRow: 0, divU: 0, u: 0 }); },
        at(s, t) { s.u = t; s.shadowN = 6 * seg(t, 0.03, 0.95, 'linear'); s.fillRow = s.shadowN; } },

      { title: 'Every row at once: S = QKᵀ', duration: 3.5,
        narration: 'Do the same for each token’s query and you fill a 6×6 table of all query–key dot products — a Gram-style matrix (p3-07) between two different sets of vectors. In one line: S = QKᵀ, shapes (6×3)(3×6) → (6×6).',
        setup(s) { Object.assign(s, { panel: 'S', mathMode: 'S', shadowN: 6, fillRow: 6, fillRows: 0, u: 0 }); },
        at(s, t) { s.u = t; s.fillRows = 6 * seg(t, 0.05, 0.8, 'linear'); } },

      { title: 'Divide by √d', duration: 5,
        narration: 'Before softmax, divide the scores by √d. Draw random queries and keys with unit-spread entries: the spread of their dot product grows like √d. Dividing by √d — <b>scaling by √d</b> — keeps scores at the same scale in d = 3 as in d = 512, so softmax is neither flat nor spiked by accident.',
        setup(s) { Object.assign(s, { panel: 'sqrt', mathMode: 'sqrt', fillRows: 6, divU: 0, u: 0 }); },
        at(s, t) { s.u = t; s.divU = seg(t, 0.55, 0.9); } },

      { title: 'Softmax turns the row into weights', duration: 5,
        narration: 'Softmax (p4-04) maps the row to <b>attention weights</b>, positive and summing to one: cat 0.49, mat 0.26, the two “the”s 0.08 each, sat 0.07, on 0.01. Lower the temperature and cat takes almost everything; raise it and the weights flatten.',
        setup(s) { Object.assign(s, { panel: 'soft', mathMode: 'soft', divU: 1, shadowN: 0, T: 1, u: 0 }); },
        at(s, t) {
          s.u = t;
          const a = seg(t, 0.15, 0.4), b = seg(t, 0.45, 0.7), c = seg(t, 0.75, 0.95);
          s.T = t < 0.42 ? lerp(1, 0.3, a) : t < 0.72 ? lerp(0.3, 3, b) : lerp(3, 1, c);
        } },

      { title: '✕ Attention picks one token', duration: 3.5, misconception: true,
        narration: '“Attention is a lookup: it selects the best match.” No — every token contributes with its weight. cat gets 0.49, but mat’s 0.26 and even on’s 0.01 are mixed in. A hard pick would be a one-hot row; attention is a soft blend, and that is what makes it differentiable (p5-06).',
        setup(s) { Object.assign(s, { panel: 'onehot', mathMode: 'onehot', T: 1, oneHot: 1, u: 0 }); },
        at(s, t) { s.u = t; s.oneHot = 1 - seg(t, 0.45, 0.85); } },

      { title: 'Output = weighted sum of values', duration: 5,
        narration: 'Now spend the weights: the new vector for sat is the <b>weighted sum</b> ' + ov + ' = Σ αⱼvⱼ (p1-02, tip to tail with scaled arrows). The purple values, each shrunk by its weight, line up head to tail; the orange result points mostly like cat’s value, with a little mat. sat now “knows” about cat.',
        setup(s) { Object.assign(s, { panel: 'sum', mathMode: 'sum', oneHot: 0, arrows: 'faint', showQ: 0, showV: 1, chainU: 0, u: 0 }); s._c0 = LA.clone(s.cam); },
        at(s, t) { s.u = t; s.chainU = seg(t, 0.15, 0.9, 'linear'); const cc = camClose(); LA.cam.tween(s, { target: s._c0.target, dist: s._c0.dist }, { target: cc.target, dist: cc.dist }, seg(t, 0, 0.2)); } },

      { title: 'All six outputs: O = AV', duration: 3.5,
        narration: 'Each row of weights times the value matrix gives each token’s new vector at once: O = softmax(QKᵀ/√d) V, shapes (6×6)(6×3) → (6×3). Three matrix products and one softmax — the whole attention layer, built from pieces you already drew.',
        setup(s) { Object.assign(s, { panel: 'AV', mathMode: 'AV', chainU: 0, showQ: 0, showV: 0.6, arrows: 'faint', showAllO: 0, u: 0 }); s.cam = camClose(); },
        at(s, t) { s.u = t; s.showAllO = seg(t, 0.1, 0.85, 'linear'); } },

      { title: 'Move a token', duration: 4, interactive: true,
        narration: 'Drag cat. As it moves away from sat’s query direction, its shadow shortens, its weight falls and the orange output swings toward mat. Attention follows similarity, live. Drag it back and the row restores itself.',
        setup(s) { Object.assign(s, { panel: 'drag', mathMode: 'drag', showAllO: 0, showQ: 1, showV: 0, arrows: 'k', shadowN: 6, chainU: 1, dragCat: true, u: 0 }); s._cat0 = X0[1].slice(); s.cam = cam3(); },
        at(s, t) { s.u = t; s.X[1] = lerpV(s._cat0, [0.2, 0.1, 1.0], LA.seg.pulse(t, 0.05, 0.95)); } },

      { title: 'A second head sees differently', duration: 4,
        narration: 'Swap W_q for a second <b>head</b>’s matrix. Same tokens, same keys and values — a different pattern: now “on” looks at the nouns around it, cat and mat. Heads are parallel copies of this picture; their outputs are stacked side by side and mixed by one more matrix.',
        setup(s) { Object.assign(s, { panel: 'headB', mathMode: 'headB', X: LA.clone(X0), dragCat: false, head: 'B', q: 3, shadowN: 0, chainU: 0, showQ: 0, u: 0 }); },
        at(s, t) { s.u = t; s.showQ = seg(t, 0.05, 0.4); s.shadowN = 6 * seg(t, 0.3, 0.7, 'linear'); } },

      { title: 'Why ML cares: the real sizes', duration: 4, mlTie: true,
        narration: 'In a transformer d is 64–128 per head and T is thousands: q and k are strips, S is a T×T heatmap, and this one row — shadows, softmax, weighted sum — is exact for every row of it. The picture stops at 3. The formula does not. p6-08 trains such weights.',
        setup(s) { Object.assign(s, { scene: 'none', panel: 'ml', mathMode: 'ml', head: 'A', q: 2, u: 0 }); },
        at(s, t) { s.u = t; } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Attention is a similarity lookup made soft. Scores are dot products of queries with keys — shadows. Softmax turns a row into weights that sum to one; √d keeps the scores tame. The output is the weighted sum of values, tip to tail.',
        cards: ['<b>Scores = shadows of keys on the query.</b> One row of S = QKᵀ.', '<b>Softmax rows → weights; /√d keeps them tame.</b> Positive, sum to one, temperature sets the sharpness.', '<b>Output = Σ αⱼvⱼ.</b> A tip-to-tail blend of the values: O = softmax(QKᵀ/√d)V.'],
        setup(s) { Object.assign(s, { scene: 'attn', view: '3d', panel: 'recap', mathMode: 'sum', head: 'A', q: 2, X: LA.clone(X0), T: 1, arrows: 'k', showQ: 1, showV: 0, shadowN: 6, chainU: 1, showAllO: 0, u: 1 }); s.cam = cam3(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Pick the query token and the head, change the temperature, switch √d scaling, or drag any token tip in 3D. Select “2D heatmaps” to see S, the weights A and the output O as tables.</p>',
      setup(s) { Object.assign(s, { scene: 'attn', view: '3d', panel: 'play', mathMode: 'play', arrows: 'k', showQ: 1, showV: 0, shadowN: 6, chainU: 1, showAllO: 0, dragCat: true, checkMode: 'play' }); s.cam = cam3(); },
      controls: [
        { type: 'preset', label: 'Query token', items: TOK.map((n, k) => ({ label: n, values: { q: k } })) },
        { type: 'slider', bind: 'T', min: 0.2, max: 20, step: 0.1, label: 'Temperature T', color: 'gold' },
        { type: 'toggle', bind: 'scale', label: 'Divide by √d' },
        { type: 'preset', label: 'Head / presets', items: [
          { label: 'Head A', values: { head: 'A' } }, { label: 'Head B', values: { head: 'B', q: 3 } },
          { label: 'Default', apply(s) { s.head = 'A'; s.q = 2; s.T = 1; s.scale = true; s.X = LA.clone(X0); } },
          { label: 'Hot (T = 0.3)', values: { T: 0.3 } }, { label: 'Flat (T = 5)', values: { T: 5 } },
        ] },
        { type: 'slider', bind: 'dIdx', min: 0, max: 4, step: 1, label: 'Random-vector dimension d', fmt: x => 'd = ' + DEXP[Math.round(x)] },
        { type: 'select', bind: 'view', label: 'View', options: [{ label: '3D arrows', value: '3d' }, { label: '2D heatmaps', value: '2d' }] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag cat so that sat attends to it with weight above 0.8.',
        setup(s) { Object.assign(s, { scene: 'attn', view: '3d', panel: 'drag', mathMode: 'drag', head: 'A', q: 2, T: 1, scale: true, arrows: 'k', showQ: 1, showV: 0, shadowN: 6, chainU: 1, dragCat: true, checkMode: 'cat' }); s.X = LA.clone(X0); s.X[1] = [0.4, 0.3, 0.3]; s.cam = cam3(); },
        judge(s) { const a = attn(s).A[2][1]; return { ok: a > 0.8, progress: clamp01(a / 0.8) }; },
        hint: 'Move cat’s key along sat’s yellow query direction and make it longer.',
        solution(s) { s.X[1] = [1.6, 0.1, -0.4]; } },
      { kind: 'manipulate',
        prompt: 'Drag the gold temperature knob so sat’s weights are nearly uniform: largest minus smallest below 0.1.',
        setup(s) { Object.assign(s, { scene: 'attn', view: '3d', panel: 'temp', mathMode: 'temp', head: 'A', q: 2, T: 1, scale: true, arrows: 'k', showQ: 1, showV: 0, shadowN: 6, chainU: 1, dragCat: false, checkMode: 'temp' }); s.X = LA.clone(X0); s.cam = cam3(); },
        judge(s) { const a = attn(s).A[2]; const d = Math.max(...a) - Math.min(...a); return { ok: d < 0.1, progress: clamp01(1 - (d - 0.1) / 0.4) }; },
        hint: 'Heat it up: a higher T flattens softmax.',
        solution(s) { s.T = 15; } },
      { kind: 'entry',
        prompt: 'Type the softmax weights for three scores (2, 0, 0) with T = 1 and no scaling.',
        bind: 'guessA', answer: [0.787, 0.107, 0.107], tol: LA.TOL.entry, label: 'α', colors: 'gold',
        setup(s) { Object.assign(s, { scene: 'none', panel: 'mini', mathMode: 'mini', guessA: [0.33, 0.33, 0.34], checkMode: 'mini' }); } },
    ],
  });

  // ---------- 3D scene ----------
  function drawScene(g, s, R, ctx) {
    const q = s.q;
    g.grid(null, { range: 2, alpha: 0.35 });
    g.axes({ length: 1.8, labels: false });
    const n = s.X.length;
    // token / key / value arrows
    for (let j = 0; j < n; j++) {
      const x = s.X[j];
      if (s.arrows === 'x') g.vector(x, { color: 'data', width: 2.2, head: 9, label: TOK[j], labelSize: 12, labelFont: 'ui', alpha: j === q ? 1 : 0.85 });
      else if (s.arrows === 'k') g.vector(R.K[j], { color: 'teal', width: 2.2, head: 9, label: TOK[j], labelSize: 12, labelFont: 'ui' });
      else if (s.arrows === 'faint') g.vector(R.K[j], { color: 'teal', width: 1.4, head: 7, alpha: 0.3, label: TOK[j], labelColor: 'muted', labelSize: 11, labelFont: 'ui' });
      if (s.showV > 0) g.vector(R.V[j], { color: 'w', width: 1.4, head: 7, alpha: 0.3 * s.showV });
    }
    // the query (drawn at half length)
    const qd = N.scale(R.Q[q], QS);
    if (s.showQ > 0) g.vector(N.scale(qd, Math.min(1, s.showQ * 1.2)), { color: 'v', width: 3, alpha: Math.min(1, s.showQ * 2), label: s.showQ > 0.6 ? 'q' : undefined, labelSize: 15 });
    if (s.showQ > 0.6) g.hud('yellow q drawn at ½ length', g.P.vp.x + 10, g.P.vp.y + g.P.vp.h - 18, { size: 11, color: 'muted' });
    // shadows of the keys on the query line
    if (s.shadowN > 0 && s.showQ > 0.3) {
      const qh = N.normalize(R.Q[q]);
      g.lineThrough([0, 0, 0], qh, { len: 2.6, color: 'v', alpha: 0.25, width: 1 });
      for (let j = 0; j < n; j++) {
        const f = clamp01(s.shadowN - j); if (f <= 0) continue;
        const k = R.K[j], foot = N.scale(qh, N.dot(k, qh));
        g.line(k, lerpV(k, foot, f), { color: 'residual', dashed: true, width: 1.1 });
        if (f > 0.5) g.line([0, 0, 0], foot, { color: 'mag', width: j === 1 ? 4 : 2.5, alpha: 0.8 });
        if (f > 0.95) g.point(foot, { r: 3.5, color: 'mag' });
      }
    }
    // weighted sum, tip to tail (heaviest first)
    if (s.chainU > 0) {
      const a = R.A[q];
      const order = a.map((w, j) => j).sort((i, j) => a[j] - a[i]);
      let o = [0, 0, 0];
      order.forEach((j, r) => {
        const f = clamp01(s.chainU * 6 - r); if (f <= 0) return;
        const piece = N.scale(R.V[j], a[j] * f);
        g.vector(o, N.add(o, piece), { color: 'w', width: 2.6, head: 8, label: r < 2 && f > 0.9 ? TOK[j] : undefined, labelSize: 11, labelFont: 'ui', labelAt: 0.5 });
        o = N.add(o, piece);
      });
      if (s.chainU > 0.95) g.vector(R.O[q], { color: 'b', width: 3.2, label: 'o', labelSize: 15 });
    }
    if (s.showAllO > 0) {
      for (let j = 0; j < n; j++) { const f = clamp01(s.showAllO * 6 - j); if (f > 0) g.vector(N.scale(R.O[j], f), { color: 'b', width: 2.4, head: 8, label: f > 0.9 ? 'o' + ['₁', '₂', '₃', '₄', '₅', '₆'][j] : undefined, labelSize: 12 }); }
    }
    if (s.dragCat) {
      g.handle('cat', s.X[1], (p, st) => { st.X[1] = p.slice(); }, { color: 'teal' });
      if (s.checkMode === 'play') for (let j = 0; j < n; j++) if (j !== 1) g.handle('tok' + j, s.X[j], (p, st) => { st.X[j] = p.slice(); }, { color: 'teal' });
    }
  }

  // ---------- panels ----------
  function sentence(g, s, x, y, hi) {
    let cx = x;
    TOK.forEach((t, j) => {
      const w = 8 * t.length + 12;
      rect(g, cx, y - 10, w - 4, 20, C.alpha(j === hi ? C.v : C.data, j === hi ? 0.45 : 0.18));
      txt(g, t, cx + (w - 4) / 2, y, { anchor: 'center', size: 12, color: j === hi ? 'v' : 'text' });
      cx += w;
    });
  }
  function drawPanel(g, s, R) {
    const S = g.size, x0 = 16, y0 = 40, pw = PX * S.w - 2 * x0;
    const p = s.panel, u = s.u, q = s.q;
    const cw = Math.max(22, Math.min(30, (pw - 36) / 6));
    const tx = x0 + 34;
    if (s.view === '2d' && p === 'play') { drawHeatmaps(g, s, R); return; }
    if (p === 'sentence') {
      sentence(g, s, x0, y0, 2);
      table(g, s.X, tx, y0 + 50, { cw: 40, ch: 20, range: [-1.2, 1.2], rowLabels: TOK, hiRow: 2, title: 'X: one row per token' });
      txt(g, 'two “the”s differ by ±0.1', x0, y0 + 50 + 6 * 20 + 20, { size: 11.5, color: 'muted' });
      return;
    }
    if (p === 'maps') {
      sentence(g, s, x0, y0, 2);
      const mw = 22;
      [['W_q', WQ, 'v'], ['W_k', WK, 'teal'], ['W_v', WV, 'w']].forEach((m, k) => {
        const mx = x0 + k * (3 * mw + 10);
        table(g, m[1], mx, y0 + 60, { cw: mw, ch: 18, range: [-3.6, 3.6], size: 8.5, title: m[0], titleColor: m[2] });
      });
      txt(g, 'q = W_q x   (sat’s query)', x0, y0 + 150, { color: 'v' });
      txt(g, 'k = W_k x = x   (W_k = I here)', x0, y0 + 170, { color: 'teal' });
      txt(g, 'v = W_v x   (faint purple)', x0, y0 + 190, { color: 'w' });
      return;
    }
    if (p === 'scores' || p === 'S' || p === 'sqrt') {
      const div = p === 'sqrt' ? lerp(1, Math.sqrt(3), s.divU) : 1;
      const M = R.S.map(r => r.map(v => v / div));
      const shown = p === 'scores' ? (i, j) => i === 2 && j < Math.floor(s.fillRow + 1e-9) : p === 'S' ? (i, j) => i === 2 || (i < 2 ? i : i - 1) < Math.floor(s.fillRows + 1e-9) : null;
      table(g, M, tx, y0 + 40, { cw, ch: 20, range: [-3.7, 3.7], rowLabels: TOK, colLabels: TOK, hiRow: 2, shown, title: p === 'sqrt' && s.divU > 0.5 ? 'S / √3' : 'scores S (rows: queries, columns: keys)' });
      const yb = y0 + 40 + 6 * 20 + 26;
      if (p === 'scores') txt(g, 'row sat: q_sat · k_j', x0, yb, { color: 'v' });
      if (p === 'S') {
        const ty = yb + 40, b1 = tile(g, x0, ty, 6, 3, { color: 'v' }), b2 = tile(g, x0 + b1.w + lerp(30, 4, seg(u, 0, 0.4)), ty, 3, 6, { color: 'teal' });
        txt(g, '→', b2.x + b2.w + 10, ty, { size: 15 }); tile(g, b2.x + b2.w + 30, ty, 6, 6, { color: 'gold' });
        txt(g, 'Q ≠ K: S is not symmetric', x0, ty + 50, { size: 11.5, color: 'muted' });
      }
      if (p === 'sqrt') {
        const bx = x0, by = yb + 30, bw = pw / 5, bh = 70;
        const mx = Math.max(...s.sdRaw);
        txt(g, 'std(q·k) over 400 random pairs', bx, by - 12, { size: 11.5, color: 'teal' });
        s.sdRaw.forEach((v, k) => { const h = v / mx * bh; rect(g, bx + k * bw + 4, by + bh - h, bw - 8, h, C.alpha(C.teal, 0.8)); txt(g, LA.fmt(v), bx + (k + 0.5) * bw, by + bh - h - 8, { anchor: 'center', size: 10 }); txt(g, 'd=' + DEXP[k], bx + (k + 0.5) * bw, by + bh + 10, { anchor: 'center', size: 10, color: 'muted' }); });
        if (u > 0.4) {
          const by2 = by + bh + 50;
          txt(g, 'std(q·k / √d): flat at ≈ 1 (axis 0…2)', bx, by2 - 12, { size: 11.5, color: 'gold' });
          s.sdScaled.forEach((v, k) => { const h = v / 2 * bh; rect(g, bx + k * bw + 4, by2 + bh - h, bw - 8, Math.max(1, h), C.alpha(C.gold, 0.85)); txt(g, LA.fmt(v), bx + (k + 0.5) * bw, by2 + bh - h - 8, { anchor: 'center', size: 10 }); });
          rect(g, bx, by2 + bh, pw, 1, C.line);
        }
      }
      return;
    }
    if (p === 'soft' || p === 'onehot' || p === 'recap' || p === 'drag' || p === 'temp') {
      const a = R.A[q];
      const bw = pw / 6, bh = 120, by = y0 + 30;
      if (p === 'onehot') {
        const oh = a.map((v, j) => j === 1 ? 1 : 0), mix = a.map((v, j) => lerp(v, oh[j], s.oneHot));
        wbars(g, mix, x0, by, bw, bh, { colors: mix.map(() => s.oneHot > 0.5 ? 'bad' : 'gold'), title: s.oneHot > 0.5 ? 'one-hot: “pick the best”' : 'the real weights: a blend' });
        g.marker.cross([x0 + 8, by + bh + 44], { screen: true, size: 20 });
        txt(g, '“attention selects the best match”', x0 + 22, by + bh + 44, { color: 'text', size: 12 });
        if (s.oneHot < 0.5) txt(g, 'blend: every token contributes', x0, by + bh + 74, { color: 'gold', size: 12.5 });
        return;
      }
      wbars(g, a, x0, by, bw, bh, { title: 'weights of ' + TOK[q] + ' (T = ' + LA.fmt(s.T) + ')' });
      if (p === 'soft' || p === 'recap') {
        // the probability triangle for (cat, mat, everything else)
        const vy = by + bh + 44, vh = Math.min(180, S.h - vy - 30);
        const rest = 1 - a[1] - a[5];
        g.viewport({ x: x0, y: vy, w: pw, h: vh }, { view: '2d', cam: { center: [0, 0.36], halfH: 0.72 }, border: false }, (gg) => {
          const A = [0, 0.9], B = [-0.6, -0.12], Cc = [0.6, -0.12];
          gg.polygon([A, B, Cc], { fill: 'gold', alpha: 0.08, stroke: 'muted' });
          gg.label('cat', A, { anchor: 'above', dy: -4, size: 11, color: 'muted', font: 'ui' });
          gg.label('mat', B, { anchor: 'below', dy: 4, size: 11, color: 'muted', font: 'ui' });
          gg.label('the rest', Cc, { anchor: 'below', dy: 4, size: 11, color: 'muted', font: 'ui' });
          const P = [a[1] * A[0] + a[5] * B[0] + rest * Cc[0], a[1] * A[1] + a[5] * B[1] + rest * Cc[1]];
          gg.point(P, { r: 6, color: 'gold', stroke: 'white' });
        });
      }
      if (p === 'drag') {
        txt(g, 'α(sat, cat) = ' + LA.fmt(a[1]), x0, by + bh + 46, { color: 'gold', size: 13 });
        txt(g, 'score(sat, cat) = ' + LA.fmt(R.S[2][1]), x0, by + bh + 68, { color: 'mag' });
        if (s.checkMode !== 'cat') txt(g, 'drag the teal tip of cat', x0, by + bh + 92, { color: 'muted', size: 11.5 });
      }
      if (p === 'temp') {
        const d = Math.max(...a) - Math.min(...a);
        txt(g, 'largest − smallest = ' + LA.fmt(d), x0, by + bh + 46, { color: d < 0.1 ? 'ok' : 'hud' });
        const ty = by + bh + 80;
        txt(g, 'temperature T = ' + LA.fmt(s.T) + '  (drag the knob)', x0, ty, { color: 'gold' });
        g.viewport({ x: x0, y: ty + 12, w: pw, h: 40 }, { view: '2d', cam: { center: [0, 0], halfH: 0.5 }, border: false }, (gg) => {
          const L = gg.P.bounds(), x1 = L.x0 + 0.15, x2 = L.x1 - 0.15;
          gg.line([x1, 0], [x2, 0], { color: 'muted', width: 2 });
          const tx2 = lerp(x1, x2, Math.log(s.T / 0.2) / Math.log(100));
          gg.point([tx2, 0], { r: 7, color: 'gold' });
          gg.label('0.2', [x1, 0], { anchor: 'below', dy: 6, size: 10, color: 'muted', font: 'ui' });
          gg.label('20', [x2, 0], { anchor: 'below', dy: 6, size: 10, color: 'muted', font: 'ui' });
          gg.handle('T', [tx2, 0], (pp, st) => { const f = clamp01((pp[0] - x1) / (x2 - x1)); st.T = Math.round(0.2 * Math.pow(100, f) * 100) / 100; }, { color: 'gold', always: true });
        });
      }
      return;
    }
    if (p === 'sum') {
      const a = R.A[q], order = a.map((w, j) => j).sort((i, j) => a[j] - a[i]);
      txt(g, 'o = Σ αⱼ vⱼ, heaviest first', x0, y0, { color: 'b', size: 13 });
      order.forEach((j, r) => { const on = s.chainU * 6 > r; txt(g, LA.fmt(a[j]) + ' · v(' + TOK[j] + ')', x0 + 8, y0 + 28 + r * 20, { color: on ? 'w' : 'muted' }); });
      if (s.chainU > 0.95) txt(g, 'o(sat) = (' + R.O[q].map(v => LA.fmt(v)).join(', ') + ')', x0, y0 + 28 + 6 * 20 + 16, { color: 'b' });
      txt(g, 'v(cat) = (' + R.V[1].map(v => LA.fmt(v)).join(', ') + ')', x0, y0 + 28 + 6 * 20 + 40, { color: 'muted', size: 11.5 });
      return;
    }
    if (p === 'AV') {
      const c2 = Math.max(18, Math.min(26, (pw - 30) / 9));
      const ax = x0 + 30, vx = ax + 6 * c2 + 10, ya = y0 + 30 + 6 * 20 + 34;
      table(g, R.V, vx, y0 + 30, { cw: c2, ch: 20, range: [-1, 1], size: 8.5, title: 'V' });
      table(g, R.A, ax, ya, { cw: c2, ch: 20, range: [0, 1], ramp: rampHeat, size: 8, rowLabels: TOK, title: 'A (rows sum to 1)', textColor: t => t > 0.6 ? 'bg' : 'text' });
      const upto = Math.floor(6 * s.showAllO + 1e-9);
      table(g, R.O, vx, ya, { cw: c2, ch: 20, range: [-1, 1], size: 8.5, shown: (i) => i < upto, title: '= O' });
      txt(g, '(6×6)(6×3) → (6×3)', x0, ya + 6 * 20 + 24, { font: 'mono', size: 12 });
      return;
    }
    if (p === 'headB') {
      const a = R.A[q];
      const bw = pw / 6, bh = 100, by = y0 + 30;
      wbars(g, a.map(v => v * clamp01(s.shadowN / 6)), x0, by, bw, bh, { title: 'head B: weights of “on”' });
      const RA = attn(Object.assign({}, s, { head: 'A' }));
      const c2 = Math.max(16, Math.min(24, (pw - 20) / 7));
      const yy = by + bh + 60;
      table(g, RA.O, x0, yy, { cw: c2, ch: 18, range: [-1, 1], values: false, title: 'O_A' });
      table(g, R.O, x0 + 3 * c2 + 4, yy, { cw: c2, ch: 18, range: [-1, 1], values: false, title: 'O_B' });
      txt(g, '· W_oᵀ (6×3)', x0 + 6 * c2 + 10, yy + 54, { font: 'mono', size: 11.5 });
      txt(g, '[O_A O_B] is 6×6 → 6×3', x0, yy + 6 * 18 + 22, { font: 'mono', size: 11.5, color: 'muted' });
      return;
    }
    if (p === 'ml') {
      const W = S.w, qk = s.qk64, cwS = (W - 80) / 64;
      txt(g, 'one query and one key in ℝ⁶⁴ (synthetic strips)', 24, 30, { size: 12.5 });
      ['q', 'k'].forEach((nm, k) => {
        g.strip(qk[k], { x: 40, y: 50 + k * 34, w: cwS * 64, h: 22, range: [-2.5, 2.5], ramp: rampDiv });
        txt(g, nm, 28, 61 + k * 34, { anchor: 'center', size: 14, color: k ? 'teal' : 'v', font: 'math' });
      });
      txt(g, 'q·k = ' + LA.fmt(N.dot(qk[0], qk[1])) + '   (one cell of S)', 40, 130, { color: 'gold' });
      const A = s.A12, c2 = Math.min(26, (Math.min(W, S.h - 200) - 80) / 12), hx = 40, hy = 180;
      txt(g, '12×12 attention weights A (synthetic, softmax per row)', hx, hy - 16, { size: 12.5 });
      A.forEach((row, i) => row.forEach((v, j) => rect(g, hx + j * c2, hy + i * c2, c2 - 1, c2 - 1, C.heat(clamp01(v / 0.5)))));
      rect(g, hx - 2, hy + 4 * c2 - 2, 12 * c2 + 2, 2, C.v); rect(g, hx - 2, hy + 5 * c2 - 1, 12 * c2 + 2, 2, C.v);
      txt(g, '← one row = one query’s weights', hx + 12 * c2 + 8, hy + 4.5 * c2, { size: 11.5, color: 'v' });
      txt(g, 'The picture stops at 3. The formula does not.', 24, S.h - 46, { color: 'gold', size: 13 });
      txt(g, 'see p6-08: training such weights', S.w - 20, S.h - 20, { color: 'gold', size: 12, anchor: 'right' });
      return;
    }
    if (p === 'mini') {
      const tru = N.softmax([2, 0, 0], 1), gs = s.guessA;
      const bx = 40, by = 90, bw = Math.min(110, (S.w - 100) / 3), bh = 200;
      txt(g, 'three keys with scores (2, 0, 0), T = 1, no scaling', 24, 30, { size: 13 });
      for (let j = 0; j < 3; j++) {
        const x = bx + j * bw;
        const ht = tru[j] * bh, hg = clamp01(gs[j]) * bh;
        rect(g, x + 8, by + bh - hg, bw / 2 - 10, Math.max(1, hg), C.alpha(C.gold, 0.85));
        txt(g, LA.fmt(gs[j]), x + 8 + (bw / 2 - 10) / 2, by + bh - hg - 9, { anchor: 'center', size: 11, color: 'gold' });
        txt(g, 'score ' + [2, 0, 0][j], x + bw / 2, by + bh + 14, { anchor: 'center', size: 11, color: 'muted' });
        const near = Math.abs(gs[j] - tru[j]) <= LA.TOL.entry;
        if (near) { rect(g, x + bw / 2 + 2, by + bh - ht, bw / 2 - 10, Math.max(1, ht), C.alpha(C.teal, 0.6)); }
      }
      rect(g, bx, by + bh, 3 * bw, 1, C.line);
      txt(g, 'gold: your weights · teal appears beside each one you got right', 24, by + bh + 44, { size: 11.5, color: 'muted' });
      txt(g, 'sum of your weights = ' + LA.fmt(gs[0] + gs[1] + gs[2]), 24, by + bh + 66, { size: 12, color: Math.abs(gs[0] + gs[1] + gs[2] - 1) < 0.02 ? 'ok' : 'hud' });
      return;
    }
    if (p === 'play') {
      const a = R.A[q], bw = pw / 6, by = y0 + 30;
      wbars(g, a, x0, by, bw, 110, { title: 'weights of ' + TOK[q] + ' · head ' + s.head });
      const sd = s.sdRaw[Math.round(s.dIdx)], d = DEXP[Math.round(s.dIdx)];
      txt(g, 'random q, k in d = ' + d + ':', x0, by + 170, { color: 'hud' });
      txt(g, 'std(q·k) = ' + LA.fmt(sd) + ' ≈ √d = ' + LA.fmt(Math.sqrt(d)), x0, by + 190, { color: 'teal' });
      txt(g, 'std(q·k/√d) = ' + LA.fmt(s.sdScaled[Math.round(s.dIdx)]), x0, by + 210, { color: 'gold' });
      return;
    }
  }
  function drawHeatmaps(g, s, R) {
    const S = g.size, c2 = Math.max(18, Math.min(30, (S.w - 120) / 15));
    const x0 = 60, y0 = 70;
    table(g, R.S.map(r => r.map(v => v / R.sc)), x0, y0, { cw: c2, ch: 20, range: [-3, 3], rowLabels: TOK, colLabels: TOK, hiRow: s.q, size: 8.5, title: s.scale ? 'S / √3' : 'S' });
    table(g, R.A, x0 + 6 * c2 + 30, y0, { cw: c2, ch: 20, range: [0, 1], ramp: rampHeat, colLabels: TOK, hiRow: s.q, size: 8.5, title: 'A = softmax rows (T = ' + LA.fmt(s.T) + ')', textColor: t => t > 0.6 ? 'bg' : 'text' });
    table(g, R.O, x0 + 12 * c2 + 60, y0, { cw: c2, ch: 20, range: [-1, 1], size: 8.5, title: 'O = AV', hiRow: s.q });
    const a = R.A[s.q];
    wbars(g, a, x0, y0 + 200, Math.min(60, (S.w - 120) / 6), 120, { title: 'weights of ' + TOK[s.q] });
  }
})();
